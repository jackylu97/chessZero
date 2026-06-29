#!/usr/bin/env python3
"""PROSECUTOR probe: is the dyn->value sibling-blindness a target artifact or an
architecture/value-equivalence defect?

Test 1 (value-equivalence gap on REAL positions, full 8-frame history):
  For real buffer positions, compare for each legal move a:
    V_dyn(a)  = wdl_to_scalar(prediction_value(dynamics(repr(obs), a)))
    V_repr(a) = -wdl_to_scalar(prediction_value(repr(obs_after_a)))   # opp POV negate
  Report std across siblings + corr. Uses FULL history (not zero-padded) by
  replaying buffer games through python-chess.

Test 2 (ground-truth sibling ranking):
  Among legal moves, pick a clearly-good one (Stockfish best) and a clearly-bad
  one (loses >=3 cp material / Stockfish worst). Does V_dyn rank good>bad?
  Does V_repr? This says whether EITHER path could give MCTS a usable signal.

Test 3 (unroll-target consistency): does V_dyn at k=1 match what the value head
  was TRAINED on for that transition (target_values[1])? If dyn-value tracks the
  draw target but repr-value does not, the dyn head overfit the all-draw unroll
  targets specifically.

Run on EARLY checkpoint (1000) where decisive games + value spread exist, vs late.
"""
import argparse, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, chess
from src.config import MuZeroConfig, get_config
from src.games.chess import ChessGame, _move_to_action, _action_to_move
from src.games.base import GameState
from src.model.muzero_net import MuZeroNetwork
from src.model.utils import wdl_to_scalar
from src.training.replay_buffer import ReplayBuffer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoint_health import build_network

CKDIR = "checkpoints/chess/2026_06_19_cold2_pc"


def load_decisive_positions(buf_path, game, hf, n_games=40, seed=0):
    """Return list of (obs_stack tensor CxHxW, board, stm) from DECISIVE games,
    sampling mid-game plies, with REAL 8-frame history."""
    rng = np.random.default_rng(seed)
    out = []
    rb = ReplayBuffer.__new__(ReplayBuffer)
    rb.load(buf_path, game=game)
    games = list(rb.buffer)
    decisive = [g for g in games if abs(float(getattr(g, "game_outcome", 0.0))) == 1.0]
    rng.shuffle(decisive)
    for gh in decisive[:n_games]:
        n = len(gh.actions)
        if n < 6:
            continue
        # sample a few mid-game plies
        for ply in rng.choice(range(2, n - 1), size=min(3, max(1, n - 3)), replace=False):
            ply = int(ply)
            obs = gh._stack_history(ply, hf)  # real history
            # reconstruct board by replaying actions
            board = chess.Board()
            ok = True
            for a in gh.actions[:ply]:
                mv = _action_to_move(int(a), board)
                if mv is None or mv not in board.legal_moves:
                    ok = False; break
                board.push(mv)
            if not ok or board.is_game_over():
                continue
            out.append((obs, board.copy()))
    return out


@torch.no_grad()
def run(path, game, cfg, dev, positions, sf=None):
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    hf = cfg.history_frames
    ds = cfg.draw_score
    dyn_stds, repr_stds, corrs = [], [], []
    gt_dyn_correct, gt_repr_correct, gt_n = 0, 0, 0
    for obs, board in positions:
        legal = list(board.legal_moves)
        if len(legal) < 4:
            continue
        acts = [_move_to_action(m, board.turn) for m in legal]
        o = obs.unsqueeze(0).to(dev)
        h = net.representation(o)
        # via dynamics (the MCTS leaf path)
        at = torch.tensor(acts, dtype=torch.long, device=dev)
        hn = h.expand(len(acts), *h.shape[1:]).contiguous()
        dyn, _ = net.dynamics(hn, at)
        _, vla = net.prediction(dyn)
        # dynamics child value is from CHILD's POV (opp to move) -> negate to root STM
        v_dyn = -wdl_to_scalar(vla.float(), draw_score=ds).cpu().numpy()
        # via representation of actual next position
        v_repr = []
        for m in legal:
            nb = board.copy(); nb.push(m)
            ns = GameState(board=nb, current_player=1 if nb.turn == chess.WHITE else -1)
            cur = game.to_tensor(ns)
            frames = [cur] + [torch.zeros_like(cur) for _ in range(hf - 1)]
            nobs = torch.cat(frames, 0).unsqueeze(0).to(dev)
            _, nvl = net.prediction(net.representation(nobs))
            v_repr.append(-wdl_to_scalar(nvl.float(), draw_score=ds).item())
        v_repr = np.asarray(v_repr)
        dyn_stds.append(float(np.std(v_dyn)))
        repr_stds.append(float(np.std(v_repr)))
        if v_dyn.std() > 1e-9 and v_repr.std() > 1e-9:
            corrs.append(float(np.corrcoef(v_dyn, v_repr)[0, 1]))
        # ground-truth ranking via Stockfish
        if sf is not None:
            try:
                info = sf.analyse(board, chess.engine.Limit(depth=8), multipv=len(legal))
                # map move -> sf score (STM POV, cp)
                mv_score = {}
                for entry in info:
                    pv = entry.get("pv")
                    if not pv:
                        continue
                    sc = entry["score"].pov(board.turn).score(mate_score=10000)
                    mv_score[pv[0]] = sc
                scored = [(m, mv_score.get(m)) for m in legal if mv_score.get(m) is not None]
                if len(scored) >= 2:
                    best = max(scored, key=lambda x: x[1])
                    worst = min(scored, key=lambda x: x[1])
                    if best[1] - worst[1] >= 100:  # at least 1 pawn gap
                        ib = legal.index(best[0]); iw = legal.index(worst[0])
                        gt_n += 1
                        if v_dyn[ib] > v_dyn[iw]:
                            gt_dyn_correct += 1
                        if v_repr[ib] > v_repr[iw]:
                            gt_repr_correct += 1
            except Exception:
                pass
    return dict(
        n=len(dyn_stds),
        dyn_std=float(np.mean(dyn_stds)) if dyn_stds else 0.0,
        repr_std=float(np.mean(repr_stds)) if repr_stds else 0.0,
        corr=float(np.mean(corrs)) if corrs else float("nan"),
        gt_n=gt_n,
        gt_dyn_acc=gt_dyn_correct / gt_n if gt_n else float("nan"),
        gt_repr_acc=gt_repr_correct / gt_n if gt_n else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", default="1000,6000,30000")
    ap.add_argument("--buf-step", default="1000", help="which buffer to draw decisive positions from")
    ap.add_argument("--sf", default="", help="path to stockfish binary (optional)")
    ap.add_argument("--n-games", type=int, default=40)
    args = ap.parse_args()
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = args.device
    hf = cfg.history_frames
    buf = os.path.join(CKDIR, f"checkpoint_{args.buf_step}.buf")
    print(f"Loading decisive positions from {buf} ...")
    positions = load_decisive_positions(buf, game, hf, n_games=args.n_games)
    print(f"  got {len(positions)} positions from decisive games")
    sf = None
    if args.sf:
        import chess.engine
        sf = chess.engine.SimpleEngine.popen_uci(args.sf)
    try:
        rows = []
        for st in [int(s) for s in args.steps.split(",")]:
            r = run(os.path.join(CKDIR, f"checkpoint_{st}.pt"), game, cfg, args.device, positions, sf)
            rows.append((st, r))
            print(f"\n=== step {st} (positions from buf {args.buf_step}) ===")
            print(f"  n={r['n']}  std(V_dyn)={r['dyn_std']:.4f}  std(V_repr)={r['repr_std']:.4f}  corr={r['corr']:.3f}")
            if sf is not None:
                print(f"  GT ranking (n={r['gt_n']}): V_dyn good>bad acc={r['gt_dyn_acc']:.3f}  V_repr acc={r['gt_repr_acc']:.3f}")
    finally:
        if sf is not None:
            sf.quit()


if __name__ == "__main__":
    main()
