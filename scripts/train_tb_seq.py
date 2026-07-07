"""Train a BLANK model on K-step TB SEQUENCES via the unroll — trains representation +
DYNAMICS + prediction (unlike train_tb_supervised which only trained the heads). Then the
dynamics is real, so MCTS should finally help. Tests value/policy acc + greedy + MCTS conv.

Run: PYTHONPATH=. .venv/bin/python scripts/train_tb_seq.py
"""
import os, pickle, time, numpy as np, torch, torch.nn.functional as F, chess
from src.config import get_config
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.games.chess_gpu import GpuChessGame
from src.model.muzero_net import MuZeroNetwork
from src.model.utils import scalar_transform, scalar_to_support, inverse_scalar_transform, support_to_scalar

DEV = "cuda"; cfg = get_config(os.environ.get("CFG", "chess_small")); game = ChessGame(); AS = game.action_space_size; NF = cfg.history_frames
K = 5; gg = GpuChessGame(); torch.manual_seed(0); np.random.seed(0)
# 2026-07-07 arch sweep params (defaults preserve historical behavior)
D_MODEL = int(os.environ.get("D_MODEL", str(cfg.hidden_planes)))
ATTN = os.environ.get("USE_ATTENTION", "0") == "1"
LAYERS = int(os.environ.get("ATTN_LAYERS", str(getattr(cfg, "attn_layers", 4))))
HEADS = int(os.environ.get("ATTN_HEADS", str(getattr(cfg, "attn_heads", 4))))
STEM = int(os.environ.get("HYBRID_STEM", str(getattr(cfg, "hybrid_stem_blocks", 0))))
PRED_L = int(os.environ.get("PRED_ATTN_LAYERS", str(getattr(cfg, "pred_attn_layers", 2))))
POLICY_HEAD = os.environ.get("POLICY_HEAD", cfg.policy_head_type)
SCALAR_POOL = os.environ.get("SCALAR_POOL", "conv")
STEPS = int(os.environ.get("STEPS", "12000"))
BATCH = int(os.environ.get("BATCH", "384"))
REWARD_W = float(os.environ.get("REWARD_W", "0.0"))
FC = int(os.environ.get("FC_HIDDEN", str(cfg.fc_hidden)))
OUT = os.environ.get("OUT", "checkpoints/tb5_seq_ml.pt")
GRAD_CKPT = os.environ.get("GRAD_CKPT", "0") == "1"
print("loading sequences...", flush=True)
SEQ = pickle.load(open("data/tb5_seq.pkl", "rb"))
te = pickle.load(open("data/tb5_test.pkl", "rb"))           # isolated positions for value/policy/conv eval
te_fen = [x[0] for x in te]; te_v = np.array([x[1] for x in te]); te_p = [np.array(x[2]) for x in te]
print(f"sequences {len(SEQ)}  test {len(te_fen)}", flush=True)

def encode(fens):
    st = gg.from_python_chess([chess.Board(f) for f in fens], device=DEV); obs = gg.to_tensor_batch(st)
    N, C, H, W = obs.shape
    if NF > 1: obs = torch.cat([obs, torch.zeros(N, (NF-1)*C, H, W, device=DEV)], 1)
    return obs

net = MuZeroNetwork(observation_channels=game.num_planes*NF, action_space_size=AS, hidden_planes=D_MODEL,
    num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w, input_h=game.board_size[0],
    input_w=game.board_size[1], fc_hidden=FC, value_support_size=cfg.value_support_size,
    reward_support_size=cfg.reward_support_size, reward_head_planes=8,
    action_embed_dim=cfg.action_embed_dim, use_consistency_loss=False,
    proj_hid=cfg.proj_hid, proj_out=cfg.proj_out, pred_hid=cfg.pred_hid, pred_out=cfg.pred_out,
    use_scalar_transform=cfg.use_scalar_transform, value_target_scale=cfg.value_target_scale, value_head_type="wdl",
    draw_score=0.0, policy_head_type=POLICY_HEAD, use_material_head=False,
    use_moves_left=True, moves_left_support_size=cfg.moves_left_support_size,
    moves_left_head_planes=8, moves_left_head_blocks=0,
    scalar_head_pool=SCALAR_POOL,
    use_repr_attention=ATTN, use_dyn_attention=ATTN, use_pred_attention=ATTN,
    attn_layers=LAYERS, attn_heads=HEADS, use_smolgen=True,
    pred_attn_layers=PRED_L, hybrid_stem_blocks=STEM,
    value_head_planes=8).to(DEV)
if GRAD_CKPT:
    from src.model.attention import BoardAttentionEncoder
    for m in net.modules():
        if isinstance(m, BoardAttentionEncoder):
            m.grad_checkpoint = True
ML_W = float(cfg.moves_left_loss_weight)   # 0.25 — aux moves-left CE weight
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

def pol_dense(pols):
    t = torch.zeros(len(pols), AS, device=DEV)
    for i, p in enumerate(pols):
        p = [a for a in p if a < AS]
        if p: t[i, torch.tensor(p, device=DEV)] = 1.0 / len(p)
    return t

@torch.no_grad()
def evaluate(mcts_sims=None):
    net.eval()
    vcorr = pcorr = n = 0; ml_won = []; ml_draw = []
    for s in range(0, min(len(te_fen), 8000), 1024):
        fb = te_fen[s:s+1024]; h = net.representation(encode(fb)); pl, vl = net.prediction(h)
        vcorr += (vl.argmax(1).cpu().numpy() == te_v[s:s+1024]).sum()
        mlp = inverse_scalar_transform(support_to_scalar(
            net.predict_moves_left(h), cfg.moves_left_support_size)).clamp(min=0).cpu().numpy().reshape(-1)
        vb = te_v[s:s+1024]
        ml_won.extend(mlp[vb == 0].tolist()); ml_draw.extend(mlp[vb == 1].tolist())
        soft = pl.cpu().numpy()
        for i, f in enumerate(fb):
            b = chess.Board(f); legal = [_move_to_action(m, b.turn) for m in b.legal_moves]
            if max(legal, key=lambda a: soft[i, a]) in set(te_p[s+i].tolist()): pcorr += 1
        n += len(fb)
    vacc, pacc = vcorr/n, pcorr/n
    mlw = float(np.mean(ml_won)) if ml_won else 0.0; mld = float(np.mean(ml_draw)) if ml_draw else 0.0
    def play(move_fn):
        won = [te_fen[i] for i in range(len(te_fen)) if te_v[i] == 0][:400]
        bs = [chess.Board(f) for f in won]; win = [b.turn for b in bs]; done = [False]*len(bs); mat = 0
        for ply in range(80):
            act = [i for i in range(len(bs)) if not done[i] and not bs[i].is_game_over()]
            for i in range(len(bs)):
                if not done[i] and bs[i].is_game_over():
                    done[i] = True
                    if bs[i].is_checkmate() and bs[i].turn != win[i]: mat += 1
            if not act: break
            mvs = move_fn([bs[i] for i in act])
            for j, i in enumerate(act):
                bs[i].push(mvs[j])
                if bs[i].is_checkmate():
                    done[i] = True
                    if bs[i].turn != win[i]: mat += 1
        return mat/len(won)
    def greedy(boards):
        h = net.representation(encode([b.fen() for b in boards])); pl, _ = net.prediction(h); soft = pl.cpu().numpy()
        return [max(b.legal_moves, key=lambda m: soft[j, _move_to_action(m, b.turn)]) for j, b in enumerate(boards)]
    gconv = play(greedy)
    mconv = None
    if mcts_sims:
        from src.mcts.tensor_mcts import TensorMCTS
        cfg.num_simulations = mcts_sims; cfg.moves_left_mcts = True; cfg.tb_root_probe = False
        m = TensorMCTS(net, game, cfg, device=DEV, hidden_dtype=torch.float32, select_backend="eager")
        def mmove(boards):
            obs = encode([b.fen() for b in boards])
            lm = torch.zeros(len(boards), AS, dtype=torch.bool, device=DEV)
            for j, b in enumerate(boards):
                for mv in b.legal_moves: lm[j, _move_to_action(mv, b.turn)] = True
            rd = m.run_batch_gpu(obs, lm, add_noise=False)
            ca = rd["child_actions"].cpu().numpy(); cv = rd["child_visits"].cpu().numpy(); res = []
            for j, b in enumerate(boards):
                slot = int(np.argmax(np.where(ca[j] != -1, cv[j], -1)))
                mv = _action_to_move(int(ca[j][slot]), b)
                res.append(mv if (mv is not None and mv in b.legal_moves) else list(b.legal_moves)[0])
            return res
        mconv = play(mmove)
    net.train(); return vacc, pacc, gconv, mconv, mlw, mld

print(f"params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
B = BATCH; t0 = time.time()
for step in range(1, STEPS + 1):
    bi = np.random.randint(0, len(SEQ), B); batch = [SEQ[i] for i in bi]
    # pad to K+1 plies
    obs_k = []; act_k = []; vc_k = []; pol_k = []; mask_k = []; ml_k = []
    for k in range(K+1):
        fens = [s[0][min(k, len(s[0])-1)] for s in batch]
        obs_k.append(encode(fens))
        vc_k.append(torch.tensor([s[2][min(k, len(s[2])-1)] for s in batch], device=DEV))
        pol_k.append(pol_dense([s[3][min(k, len(s[3])-1)] for s in batch]))
        ml_k.append(torch.tensor([float(s[4][min(k, len(s[4])-1)]) for s in batch], device=DEV))
        mask_k.append(torch.tensor([1.0 if k < len(s[0]) else 0.0 for s in batch], device=DEV))
        if k < K:
            act_k.append(torch.tensor([s[1][k] if k < len(s[1]) else 0 for s in batch], dtype=torch.long, device=DEV))
    h = net.representation(obs_k[0]); loss = 0.0; mloss = 1.0/(K+1)
    for k in range(K+1):
        if k == 0:
            pl, vl = net.prediction(h)
        else:
            h, rl, pl, vl = net.recurrent_inference_logits(h, act_k[k-1])
            h.register_hook(lambda g: g * 0.5)
            if REWARD_W > 0.0:
                # Reward supervision (2026-07-07): mate transition <=> the landed
                # position has plies-to-end 0. Target one-hot at +1 (mover POV);
                # everything else 0. Gives the proxy a trained mate detector.
                rew_cls = torch.where(ml_k[k] <= 0.5,
                                      torch.full_like(vc_k[k], 2),
                                      torch.full_like(vc_k[k], 1))  # support {-1,0,+1} idx
                rce = F.cross_entropy(rl, rew_cls, reduction='none') * mask_k[k]
                loss = loss + mloss * REWARD_W * rce.mean()
            # latent consistency: dynamics latent should match the real position's latent
            with torch.no_grad():
                ht = net.representation(obs_k[k])
            cons = (1 - F.cosine_similarity(h.flatten(1), ht.flatten(1), dim=1)) * mask_k[k]
            loss = loss + mloss * cons.mean()
        vce = F.cross_entropy(vl, vc_k[k], reduction='none') * mask_k[k]
        pce = -(pol_k[k] * F.log_softmax(pl, 1)).sum(1) * mask_k[k]
        ml_logits = net.predict_moves_left(h)   # (B, 2*support+1) plies-to-end logits
        ml_tgt = scalar_to_support(scalar_transform(ml_k[k]), cfg.moves_left_support_size).to(ml_logits.device)
        mlce = -(ml_tgt * F.log_softmax(ml_logits, 1)).sum(1) * mask_k[k]
        loss = loss + mloss * (vce.mean() + pce.mean() + ML_W * mlce.mean())
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        vacc, pacc, gconv, mconv, mlw, mld = evaluate(mcts_sims=(200 if step % 4000 == 0 else None))
        ms = f"  MCTS_conv(200) {mconv:.3f}" if mconv is not None else ""
        print(f"step {step:5d} ({time.time()-t0:.0f}s) loss {loss.item():.3f} | value_acc {vacc:.3f} "
              f"policy_acc {pacc:.3f} GREEDY_conv {gconv:.3f}{ms} | ml_won {mlw:.0f} ml_draw {mld:.0f}", flush=True)
import os
os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state_dict": net.state_dict()}, OUT)
print(f"saved {OUT}", flush=True)
print("DONE", flush=True)
