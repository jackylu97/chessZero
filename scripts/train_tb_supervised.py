"""Train a BLANK model purely on the 5-man tablebase (supervised: value WDL + optimal-move
policy). No self-play, no MCTS, no relabel. Tests whether a from-scratch net can learn to
evaluate AND convert <=5-man endgames from ground truth alone.

Run: PYTHONPATH=. .venv/bin/python scripts/train_tb_supervised.py
"""
import pickle, time, numpy as np, torch, torch.nn.functional as F, chess
from src.config import get_config
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.games.chess_gpu import GpuChessGame
from src.model.muzero_net import MuZeroNetwork

DEV = "cuda"; cfg = get_config("chess_small"); game = ChessGame(); AS = game.action_space_size; NF = cfg.history_frames
gg = GpuChessGame()
torch.manual_seed(0); np.random.seed(0)

def load(p):
    d = pickle.load(open(p, "rb"))
    fens = [x[0] for x in d]; vcls = np.array([x[1] for x in d], dtype=np.int64)
    pol = [np.array(x[2], dtype=np.int64) for x in d]
    return fens, vcls, pol

print("loading data...", flush=True)
tr_fen, tr_v, tr_p = load("data/tb5_train.pkl")
te_fen, te_v, te_p = load("data/tb5_test.pkl")
print(f"train {len(tr_fen)}  test {len(te_fen)}", flush=True)

def encode(fens):
    st = gg.from_python_chess([chess.Board(f) for f in fens], device=DEV)
    obs = gg.to_tensor_batch(st)
    N, C, H, W = obs.shape
    if NF > 1:
        obs = torch.cat([obs, torch.zeros(N, (NF-1)*C, H, W, device=DEV)], 1)
    return obs

net = MuZeroNetwork(observation_channels=game.num_planes*NF, action_space_size=AS, hidden_planes=cfg.hidden_planes,
    num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w, input_h=game.board_size[0],
    input_w=game.board_size[1], fc_hidden=cfg.fc_hidden, value_support_size=cfg.value_support_size,
    reward_support_size=cfg.reward_support_size, action_embed_dim=cfg.action_embed_dim, use_consistency_loss=False,
    proj_hid=cfg.proj_hid, proj_out=cfg.proj_out, pred_hid=cfg.pred_hid, pred_out=cfg.pred_out,
    use_scalar_transform=cfg.use_scalar_transform, value_target_scale=cfg.value_target_scale, value_head_type="wdl",
    draw_score=0.0, policy_head_type=cfg.policy_head_type, use_moves_left=False, use_material_head=False).to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

def fwd(obs):
    h = net.representation(obs)
    pl, vl = net.prediction(h)
    return pl, vl   # policy_logits (B,AS), value_logits (B,3) WDL

@torch.no_grad()
def policy_target(fens_b, pols_b):
    t = torch.zeros(len(fens_b), AS, device=DEV)
    for i, p in enumerate(pols_b):
        p = p[p < AS]
        if len(p): t[i, torch.tensor(p, device=DEV)] = 1.0 / len(p)
    return t

@torch.no_grad()
def evaluate():
    net.eval()
    # value + policy accuracy on test
    vcorr = pcorr = n = 0
    for s in range(0, min(len(te_fen), 8000), 1024):
        fb = te_fen[s:s+1024]; obs = encode(fb)
        pl, vl = fwd(obs)
        vpred = vl.argmax(1).cpu().numpy()
        vcorr += (vpred == te_v[s:s+1024]).sum()
        soft = pl.cpu().numpy()
        for i, f in enumerate(fb):
            b = chess.Board(f); legal = [_move_to_action(m, b.turn) for m in b.legal_moves]
            top = max(legal, key=lambda a: soft[i, a])
            pcorr += int(top in set(te_p[s+i].tolist()))
        n += len(fb)
    vacc, pacc = vcorr/n, pcorr/n
    # greedy NO-SEARCH conversion: won test positions, both sides play policy argmax, cap 80 plies
    won = [te_fen[i] for i in range(len(te_fen)) if te_v[i] == 0][:600]
    boards = [chess.Board(f) for f in won]; winner = [b.turn for b in boards]; done = [False]*len(boards); mated = 0
    for ply in range(80):
        active = [i for i, b in enumerate(boards) if not done[i]]
        if not active: break
        obs = encode([boards[i].fen() for i in active]); pl, _ = fwd(obs); soft = pl.cpu().numpy()
        for j, i in enumerate(active):
            b = boards[i]
            if b.is_game_over():
                done[i] = True
                if b.is_checkmate() and b.turn != winner[i]: mated += 1
                continue
            legal = list(b.legal_moves)
            am = max(legal, key=lambda m: soft[j, _move_to_action(m, b.turn)])
            b.push(am)
            if b.is_checkmate():  # winner just delivered mate
                done[i] = True
                if b.turn != winner[i]: mated += 1
    conv = mated/len(won)
    net.train()
    return vacc, pacc, conv

print(f"params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
B = 1024; idx = np.arange(len(tr_fen)); t0 = time.time()
for step in range(1, 8001):
    bi = np.random.randint(0, len(tr_fen), B)
    fb = [tr_fen[i] for i in bi]; obs = encode(fb)
    pl, vl = fwd(obs)
    vt = torch.tensor(tr_v[bi], device=DEV)
    pt = policy_target(fb, [tr_p[i] for i in bi])
    vloss = F.cross_entropy(vl, vt)
    ploss = -(pt * F.log_softmax(pl, 1)).sum(1).mean()
    loss = vloss + ploss
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        vacc, pacc, conv = evaluate()
        print(f"step {step:5d} ({time.time()-t0:.0f}s)  vloss {vloss.item():.3f} ploss {ploss.item():.3f} | "
              f"value_acc {vacc:.3f}  policy_acc {pacc:.3f}  GREEDY_CONVERSION {conv:.3f}", flush=True)

import os
os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state_dict": net.state_dict()}, "checkpoints/tb5_supervised.pt")
print("saved checkpoints/tb5_supervised.pt", flush=True)

# MCTS conversion: won test positions, BOTH sides play model + search, cap 80 plies.
@torch.no_grad()
def mcts_convert(sims):
    from src.mcts.tensor_mcts import TensorMCTS
    cfg.num_simulations = sims; cfg.moves_left_mcts = False; cfg.tb_root_probe = False
    mcts = TensorMCTS(net, game, cfg, device=DEV, hidden_dtype=torch.float32, select_backend="eager")
    net.eval()
    won = [te_fen[i] for i in range(len(te_fen)) if te_v[i] == 0][:512]
    boards = [chess.Board(f) for f in won]; winner = [b.turn for b in boards]; done = [False]*len(boards); mated = 0
    for ply in range(80):
        active = [i for i, b in enumerate(boards) if not (done[i] or boards[i].is_game_over())]
        for i in range(len(boards)):
            if not done[i] and boards[i].is_game_over():
                done[i] = True
                if boards[i].is_checkmate() and boards[i].turn != winner[i]: mated += 1
        if not active: break
        obs = encode([boards[i].fen() for i in active])
        lm = torch.zeros(len(active), AS, dtype=torch.bool, device=DEV)
        for j, i in enumerate(active):
            for m in boards[i].legal_moves: lm[j, _move_to_action(m, boards[i].turn)] = True
        rd = mcts.run_batch_gpu(obs, lm, add_noise=False)
        ca = rd["child_actions"].cpu().numpy(); cv = rd["child_visits"].cpu().numpy()
        for j, i in enumerate(active):
            slot = int(np.argmax(np.where(ca[j] != -1, cv[j], -1)))
            mv = _action_to_move(int(ca[j][slot]), boards[i])
            if mv is None or mv not in boards[i].legal_moves: mv = list(boards[i].legal_moves)[0]
            boards[i].push(mv)
            if boards[i].is_checkmate():
                done[i] = True
                if boards[i].turn != winner[i]: mated += 1
    return mated/len(won)

for sims in (100, 400):
    c = mcts_convert(sims)
    print(f"MCTS_CONVERSION ({sims} sims, both sides search): {c:.3f}", flush=True)
print("DONE", flush=True)
