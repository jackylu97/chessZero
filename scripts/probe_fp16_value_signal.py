"""fp16 value-signal probe.

Does fp16 (tensor_mcts_hidden_dtype='float16' + tensor_mcts_amp_dtype='float16')
round away the within-position value signal MCTS needs to rank sibling moves?

Empirical comparison on real buffer positions (checkpoint_6000):
  (1) network value path: WDL probs + value scalar, fp32 vs fp16-autocast.
  (2) node_hidden fp16 STORAGE round-trip quant error.
  (3) within-position SIBLING value spread (1-ply recurrent) fp32 vs fp16.
  (4) full TensorMCTS root_value + root child-Q spread fp32 vs fp16.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.model.utils import support_to_scalar, inverse_scalar_transform
from scripts.eval_checkpoint_health import build_network

torch.serialization.add_safe_globals([MuZeroConfig])
DEV = "cuda"
CKPT = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
BUF = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"
N_POS = 24

cfg = get_config("chess_small"); cfg.device = DEV
game = ChessGame()
HF = cfg.history_frames
ckpt = torch.load(CKPT, map_location=DEV, weights_only=True)
net = build_network(ckpt, game, cfg, DEV)
net.eval()
print(f"value_head_type={getattr(net,'value_head_type','?')} draw_score={getattr(net,'draw_score','?')} "
      f"value_support_size={net.value_support_size} hidden_planes={cfg.hidden_planes} "
      f"num_sims={cfg.num_simulations} sample_k={cfg.sample_k}")

from src.training.replay_buffer import ReplayBuffer
rb = ReplayBuffer(cfg)
rb.load(BUF, game=game)
print(f"buffer games={len(rb.buffer)}")

rng = np.random.default_rng(0)
positions = []  # (obs[C,H,W], legal_actions list)
for gi in rng.permutation(len(rb.buffer)):
    g = rb.buffer[int(gi)]
    n = len(g.observations)
    if n < 6:
        continue
    ply = int(rng.integers(2, n - 1))
    obs = g._stack_history(ply, HF)
    legal = None
    if g.legal_actions_list and ply < len(g.legal_actions_list):
        legal = list(g.legal_actions_list[ply])
    positions.append((obs, legal))
    if len(positions) >= N_POS:
        break
print(f"collected {len(positions)} positions; with legal-lists: "
      f"{sum(1 for _,l in positions if l)}")

obs_batch = torch.stack([o.to(DEV).float() for o, _ in positions], 0)
print("obs_batch", tuple(obs_batch.shape), obs_batch.dtype)


def value_from_logits(value_logits):
    if getattr(net, "value_head_type", "support") == "wdl":
        probs = torch.softmax(value_logits.float(), dim=-1)
        v = probs[..., 0] - probs[..., 2] + getattr(net, "draw_score", 0.0) * probs[..., 1]
        return v, probs
    scalar = support_to_scalar(value_logits.float(), net.value_support_size)
    if net.use_scalar_transform:
        scalar = inverse_scalar_transform(scalar)
    elif net.value_target_scale != 1.0:
        scalar = scalar / net.value_target_scale
    return scalar, torch.softmax(value_logits.float(), dim=-1)


# ===== (1) initial_inference fp32 vs fp16 =====
print("\n==== (1) initial_inference: fp32 vs fp16 autocast ====")
with torch.no_grad():
    h32, pl32, vlog32 = net.initial_inference_logits(obs_batch)
    v32, probs32 = value_from_logits(vlog32)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        h16, pl16, vlog16 = net.initial_inference_logits(obs_batch)
    v16, probs16 = value_from_logits(vlog16)

dv = (v16 - v32).abs()
print(f"value scalar |fp16-fp32|: mean={dv.mean():.6f} max={dv.max():.6f}")
print(f"value scalar fp32: std(across pos)={v32.std():.6f} range=[{v32.min():.4f},{v32.max():.4f}]")
print(f"value scalar fp16: std(across pos)={v16.std():.6f} range=[{v16.min():.4f},{v16.max():.4f}]")
print(f"WDL probs |fp16-fp32|: mean={(probs16-probs32).abs().mean():.6f} max={(probs16-probs32).abs().max():.6f}")
print(f"value logits |fp16-fp32|: mean={(vlog16.float()-vlog32).abs().mean():.6f} "
      f"max={(vlog16.float()-vlog32).abs().max():.6f}")
print(f"hidden latent |fp16-fp32|: mean={(h16.float()-h32).abs().mean():.6f} "
      f"max={(h16.float()-h32).abs().max():.6f}  (latent std={h32.std():.4f})")

print("\n-- value scalar decoded INSIDE fp16 autocast (MCTS-faithful) --")
with torch.no_grad():
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        _, _, vraw16 = net.initial_inference(obs_batch)
    v16_in = vraw16.view(-1).float()
    _, _, vraw32 = net.initial_inference(obs_batch)
    v32_p = vraw32.view(-1).float()
print(f"value (in-autocast decode) |fp16-fp32|: mean={(v16_in-v32_p).abs().mean():.6f} "
      f"max={(v16_in-v32_p).abs().max():.6f}")
print(f"  fp32-path std={v32_p.std():.6f}  fp16-path std={v16_in.std():.6f}")
print(f"  decoded value dtype under autocast = {vraw16.dtype}")


# ===== (2) node_hidden fp16 STORAGE round-trip =====
print("\n==== (2) node_hidden fp16 storage round-trip ====")
with torch.no_grad():
    h32_store = h32.clone()
    h16_store = h32.to(torch.float16).to(torch.float32)
    pl_a, vlog_a = net.prediction(h32_store)
    pl_b, vlog_b = net.prediction(h16_store)
    va, _ = value_from_logits(vlog_a)
    vb, _ = value_from_logits(vlog_b)
print(f"latent fp16-storage quant error: mean={(h16_store-h32_store).abs().mean():.6e} "
      f"max={(h16_store-h32_store).abs().max():.6e}")
print(f"value after storage round-trip |fp16store-fp32|: mean={(vb-va).abs().mean():.6f} "
      f"max={(vb-va).abs().max():.6f}")
print(f"value std fp32-stored={va.std():.6f} fp16-stored={vb.std():.6f}")


# ===== (3) SIBLING value spread within position =====
print("\n==== (3) within-position SIBLING value spread (1-ply recurrent) ====")
sib_rows = []
for pi, (obs, legal) in enumerate(positions):
    if not legal:
        continue
    o = obs.to(DEV).float().unsqueeze(0)
    with torch.no_grad():
        h, _, _ = net.initial_inference(o)
        acts = torch.tensor(legal, device=DEV, dtype=torch.long)
        hrep = h.expand(len(legal), -1, -1, -1).contiguous()
        _, _, _, vcl32 = net.recurrent_inference_logits(hrep, acts)
        vc32, _ = value_from_logits(vcl32)
        hrep16 = hrep.to(torch.float16).to(torch.float32)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            _, _, _, vcl16 = net.recurrent_inference_logits(hrep16.to(torch.float16), acts)
        vc16, _ = value_from_logits(vcl16)
    vc32 = vc32.float(); vc16 = vc16.float()
    # spearman-ish rank agreement of siblings
    order32 = torch.argsort(vc32); order16 = torch.argsort(vc16)
    rank_match = float((order32 == order16).float().mean())
    sib_rows.append((len(legal), float(vc32.std()), float(vc16.std()),
                     float((vc16 - vc32).abs().mean()),
                     float(vc32.max() - vc32.min()), float(vc16.max() - vc16.min()),
                     rank_match))
    if pi < 8:
        print(f"  pos{pi} nlegal={len(legal):2d} | sibV std fp32={vc32.std():.6f} "
              f"fp16={vc16.std():.6f} | range fp32={vc32.max()-vc32.min():.5f} "
              f"fp16={vc16.max()-vc16.min():.5f} | |Δ|mean={(vc16-vc32).abs().mean():.6f} "
              f"| argbest f32={int(vc32.argmax())} f16={int(vc16.argmax())}")
if sib_rows:
    arr = np.array(sib_rows)
    print(f"\n  SIBLING std fp32 mean={arr[:,1].mean():.6f}  fp16 mean={arr[:,2].mean():.6f}  "
          f"ratio fp16/fp32={arr[:,2].mean()/max(arr[:,1].mean(),1e-12):.3f}")
    print(f"  sibling-value |fp16-fp32| per-move mean={arr[:,3].mean():.6f}")
    print(f"  NOISE/SIGNAL = |Δ|mean / sibStd_fp32 = "
          f"{arr[:,3].mean()/max(arr[:,1].mean(),1e-12):.3f}")
    print(f"  sibling RANGE fp32 mean={arr[:,4].mean():.6f} fp16 mean={arr[:,5].mean():.6f}")
    print(f"  argmax(best sibling) preserved fp16==fp32 in "
          f"{int((arr[:,0]>0).sum())} positions; per-pos full-order match mean={arr[:,6].mean():.3f}")


# ===== (4) full TensorMCTS root_value + child-Q spread =====
print("\n==== (4) full TensorMCTS root_value + root child-Q spread fp32 vs fp16 ====")
from src.mcts.tensor_mcts import TensorMCTS
sub = [(o, l) for o, l in positions if l][:12]
if sub:
    N = len(sub)
    A = game.action_space_size
    obs_b = torch.stack([o.to(DEV).float() for o, _ in sub], 0)
    legal_mask = torch.zeros(N, A, dtype=torch.bool, device=DEV)
    for i, (_, l) in enumerate(sub):
        legal_mask[i, torch.tensor(l, device=DEV)] = True

    def run_mcts_obj(hidden_dtype, amp_dtype, seed=123):
        torch.manual_seed(seed); np.random.seed(seed)
        mcts = TensorMCTS(
            network=net, game=game, config=cfg, device=torch.device(DEV),
            hidden_dtype=hidden_dtype, amp_dtype=amp_dtype,
            compile=False, select_backend="eager",
        )
        out = mcts.run_batch_gpu(obs_b, legal_mask, add_noise=False)
        return mcts, out

    def child_q_spread(mcts):
        vis = mcts.child_visits[:, 0, :].float()
        vsum = mcts.child_value_sum[:, 0, :].float()
        q = vsum / vis.clamp(min=1)
        spreads, qranges, bestslot = [], [], []
        for i in range(vis.shape[0]):
            valid = (mcts.child_actions[i, 0, :] != -1) & (vis[i] > 0)
            if int(valid.sum()) >= 2:
                qv = q[i][valid]
                spreads.append(float(qv.std()))
                qranges.append(float(qv.max() - qv.min()))
                # best action by Q
                idx = torch.nonzero(valid).view(-1)
                bestslot.append(int(mcts.child_actions[i, 0, idx[qv.argmax()]]))
            else:
                bestslot.append(-1)
        return spreads, qranges, bestslot

    m32, out32 = run_mcts_obj(torch.float32, None)
    m16, out16 = run_mcts_obj(torch.float16, torch.float16)
    rv32 = out32["root_value"].float().cpu().numpy()
    rv16 = out16["root_value"].float().cpu().numpy()
    print(f"root_value fp32: {np.round(rv32,4)}")
    print(f"root_value fp16: {np.round(rv16,4)}")
    print(f"root_value |fp16-fp32|: mean={np.abs(rv16-rv32).mean():.6f} max={np.abs(rv16-rv32).max():.6f}")
    print(f"root_value range fp32=[{rv32.min():.3f},{rv32.max():.3f}] fp16=[{rv16.min():.3f},{rv16.max():.3f}]")

    sp32, qr32, bs32 = child_q_spread(m32); sp16, qr16, bs16 = child_q_spread(m16)
    print(f"\nroot child-Q sibling spread (std within position):")
    print(f"  fp32: mean={np.mean(sp32):.6f} per-pos={np.round(sp32,5)}")
    print(f"  fp16: mean={np.mean(sp16):.6f} per-pos={np.round(sp16,5)}")
    print(f"  ratio fp16/fp32 = {np.mean(sp16)/max(np.mean(sp32),1e-12):.3f}")
    print(f"  child-Q RANGE fp32 mean={np.mean(qr32):.6f} fp16 mean={np.mean(qr16):.6f}")
    print(f"  best-Q action fp32={bs32}")
    print(f"  best-Q action fp16={bs16}")
    print(f"  best-Q action agreement = {np.mean([a==b for a,b in zip(bs32,bs16)]):.3f}")

print("\nDONE")
