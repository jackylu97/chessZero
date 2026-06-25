"""Does the NET reproduce the good TB targets WITHOUT the probe? Isolates the
two failure candidates behind 'converts with probe, hangs queen without':

  (A) PRIOR under-fit  : net's raw policy prior doesn't point at the DTZ-optimal
                         move even though the stored TARGET does (97% optimal).
  (B) VALUE flat       : net's value head gives ~equal value to a DTZ-optimal
                         child and a slow/shuffle child -> value-driven MCTS
                         can't prefer progress, washing out a correct prior.
                         (corr(net_value_of_child, -dtz_after) ~ 0  => flat.)

Replays buffer games (faithful history), and at every clean-win (wdl=2) TB root
queries the net (CPU, doesn't touch the live GPU run). Compares:
  net prior argmax  vs  Syzygy DTZ-optimal   (fit of policy)
  net root value    vs  stored root_value    (fit of value)
  value spread / corr across win-preserving children   (the missing mechanism)

Run: PYTHONPATH=. .venv/bin/python scripts/probe_net_tb_eval.py \
        --buf <ckpt.buf> --checkpoint <ckpt.pt>
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess, chess.syzygy

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer, stack_with_history
from scripts.eval_checkpoint_health import build_network


@torch.no_grad()
def net_eval(net, obs_batch, dev):
    obs = torch.stack(obs_batch).to(dev)
    hidden, logits, value = net.initial_inference(obs)
    return logits.cpu().numpy(), value.cpu().numpy().reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tb", default="data/syzygy")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-pieces", type=int, default=5)
    ap.add_argument("--max-games", type=int, default=600)
    ap.add_argument("--max-roots", type=int, default=1200)
    ap.add_argument("--child-roots", type=int, default=300, help="subsample for value-spread test")
    args = ap.parse_args()

    dev = args.device
    game = ChessGame(); cfg = get_config("chess_small"); HF = getattr(cfg, "history_frames", 8)
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev); net.eval()
    tb = chess.syzygy.open_tablebase(args.tb)

    rb = ReplayBuffer(max_size=10_000_000); rb.load(args.buf, game=game)
    sp = [g for g in rb.buffer if not getattr(g, "external_values", [])][: args.max_games]
    print(f"net step {ckpt.get('step','?')} | {len(sp)} games | device {dev}\n")

    roots = []  # (obs, board_fen, frames_copy, stored_rv)
    for g in sp:
        actions = list(getattr(g, "actions", []) or [])
        rvs = getattr(g, "root_values", None)
        if not actions:
            continue
        b = chess.Board(getattr(g, "start_fen", None) or chess.STARTING_FEN)
        frames = []
        for t, a in enumerate(actions):
            cur = game.to_tensor(game.reset_from_fen(b.fen()))
            if len(b.piece_map()) <= args.max_pieces:
                try: wdl = tb.probe_wdl(b)
                except Exception: wdl = None
                if wdl == 2:
                    obs = stack_with_history(cur, frames, HF)
                    rv = float(rvs[t]) if rvs is not None and t < len(rvs) else float("nan")
                    roots.append((obs, b.fen(), list(frames), rv))
            frames.append(cur)
            mv = _action_to_move(int(a), b)
            if mv is None or mv not in b.legal_moves:
                break
            b.push(mv)
        if len(roots) >= args.max_roots:
            break
    print(f"collected {len(roots)} clean-win TB roots\n")

    # ---- (A) prior fit + (B-coarse) root value fit ----
    prior_opt = prior_pres = 0; n = 0
    netv, storedv = [], []
    B = 256
    for i in range(0, len(roots), B):
        chunk = roots[i:i+B]
        logits, value = net_eval(net, [c[0] for c in chunk], dev)
        for j, (obs, fen, frames, rv) in enumerate(chunk):
            b = chess.Board(fen)
            legal = list(b.legal_moves)
            # map legal moves -> action ids
            la = {}
            for mv in legal:
                from src.games.chess import _move_to_action
                la[_move_to_action(mv, b.turn)] = mv
            lg = logits[j]
            mask = np.full(lg.shape, -1e9);
            for ai in la:
                if 0 <= ai < lg.shape[0]: mask[ai] = lg[ai]
            am = int(np.argmax(mask)); mv = la.get(am)
            if mv is None:
                continue
            n += 1
            b.push(mv)
            try:
                keep = b.is_checkmate() or (tb.probe_wdl(b) < 0)
                d_am = 0 if b.is_checkmate() else abs(tb.probe_dtz(b))
            except Exception:
                keep = False; d_am = None
            b.pop()
            if keep: prior_pres += 1
            # DTZ-optimal?
            best = None
            for ai, m2 in la.items():
                b.push(m2)
                try:
                    k2 = b.is_checkmate() or (tb.probe_wdl(b) < 0)
                    d2 = 0 if b.is_checkmate() else abs(tb.probe_dtz(b))
                except Exception:
                    k2 = False; d2 = None
                b.pop()
                if k2 and d2 is not None:
                    best = d2 if best is None else min(best, d2)
            if keep and d_am is not None and best is not None and d_am == best:
                prior_opt += 1
            netv.append(float(value[j]));
            if np.isfinite(rv): storedv.append((float(value[j]), rv))

    netv = np.array(netv)
    print(f"=== (A) PRIOR fit  (target was 99.8% preserve / 97.2% DTZ-optimal) ===")
    print(f"  net prior argmax PRESERVES win:   {prior_pres/max(1,n):.1%}")
    print(f"  net prior argmax is DTZ-OPTIMAL:  {prior_opt/max(1,n):.1%}   (n={n})")
    print(f"\n=== root VALUE fit ===")
    print(f"  net root value:   mean {netv.mean():+.3f}  median {np.median(netv):+.3f}")
    if storedv:
        sv = np.array(storedv); c = np.corrcoef(sv[:,0], sv[:,1])[0,1]
        print(f"  net vs stored root_value corr: {c:+.3f}  (stored mean {sv[:,1].mean():+.3f})")

    # ---- (B) value-head flatness across win-preserving children ----
    corrs, spreads = [], []
    sub = roots[:args.child_roots]
    for obs, fen, frames, rv in sub:
        b = chess.Board(fen)
        child_obs, child_dtz = [], []
        nf = frames + [game.to_tensor(game.reset_from_fen(b.fen()))]
        for mv in b.legal_moves:
            b.push(mv)
            try:
                keep = b.is_checkmate() or (tb.probe_wdl(b) < 0)
                d = 0 if b.is_checkmate() else abs(tb.probe_dtz(b))
            except Exception:
                keep = False; d = None
            if keep and d is not None and not b.is_checkmate():
                cur = game.to_tensor(game.reset_from_fen(b.fen()))
                child_obs.append(stack_with_history(cur, nf, HF)); child_dtz.append(d)
            b.pop()
        if len(child_obs) >= 3:
            _, cv = net_eval(net, child_obs, dev)
            mover_v = -cv  # child is opponent to move; mover POV = -child value
            d = np.array(child_dtz, dtype=np.float64)
            if d.std() > 0 and mover_v.std() > 0:
                corrs.append(np.corrcoef(mover_v, -d)[0,1])  # +1 => lower dtz -> higher value
            spreads.append(float(mover_v.max() - mover_v.min()))
    tb.close()
    print(f"\n=== (B) VALUE-HEAD progress signal across win-preserving children ===")
    print(f"  positions tested: {len(spreads)}")
    if corrs:
        print(f"  corr(net child value, -DTZ):  mean {np.mean(corrs):+.3f}  median {np.median(corrs):+.3f}")
        print(f"    (>+0.3 = value encodes progress; ~0 = FLAT -> search can't convert)")
    if spreads:
        print(f"  value spread (max-min) over winning children: mean {np.mean(spreads):.3f}")
        print(f"    (tiny spread => all winning moves look equal to the value head)")


if __name__ == "__main__":
    main()
