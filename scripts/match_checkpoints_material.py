"""Head-to-head match between two checkpoints WITH per-game material tracking.

Same paired/color-swapped design as match_checkpoints.py, plus it records for
every game the peak and final material margin (white minus black, in pawns,
P/N=3/B=3/R=5/Q=9) re-expressed from A's perspective. The point is to test the
hypothesis "the newer net wins the midgame but fails to convert": among DRAWN
games, how often was one side materially winning, and what is each net's
conversion rate once it reaches a material edge.

Run:
  .venv/bin/python scripts/match_checkpoints_material.py \
      --checkpoint-a checkpoints/chess/<run>/checkpoint_40000.pt \
      --checkpoint-b checkpoints/chess/<run>/checkpoint_31000.pt \
      --game chess_small --games 200 --sims 160 --device cuda \
      --dump /tmp/.../unconverted.txt
"""
import argparse, math, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.mcts.mcts import BatchedMCTS, select_action
from src.training.replay_buffer import stack_with_history
from scripts.eval_checkpoint_health import build_network


PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9}


def material_white(board):
    """Material margin in pawns, white minus black (kings excluded)."""
    m = 0
    for pt, val in PIECE_VAL.items():
        m += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return m


def load_net(path, game, cfg, dev):
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    return build_network(ckpt, game, cfg, dev), ckpt.get("step", "?")


def gen_openings(game, n_openings, plies, seed):
    rng = random.Random(seed)
    openings = []
    for _ in range(n_openings):
        s = game.reset(); seq = []
        for _ in range(plies):
            if s.done:
                break
            a = rng.choice(game.legal_actions(s))
            seq.append(a); s, _, _ = game.step(s, a)
        openings.append(seq)
    return openings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-a", "-a", required=True, help="model A (e.g. newer)")
    ap.add_argument("--checkpoint-b", "-b", required=True, help="model B (e.g. older)")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=250)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--game", default="chess")
    ap.add_argument("--dump", default=None, help="write per-game material table here")
    args = ap.parse_args()

    dev = args.device
    game = ChessGame()
    cfg = get_config(args.game); cfg.device = dev
    cfg.num_simulations = args.sims
    n_frames = getattr(cfg, "history_frames", 1)

    netA, stepA = load_net(args.checkpoint_a, game, cfg, dev)
    netB, stepB = load_net(args.checkpoint_b, game, cfg, dev)
    mcts = {"A": BatchedMCTS(netA, game, cfg, dev), "B": BatchedMCTS(netB, game, cfg, dev)}

    n_pairs = args.games // 2
    openings = gen_openings(game, n_pairs, args.opening_plies, args.seed)

    states, obs_hist, white_net, opening_seq, mc, done, outcome = [], [], [], [], [], [], []
    for k in range(n_pairs):
        for wn in ("A", "B"):
            states.append(game.reset()); obs_hist.append([]); white_net.append(wn)
            opening_seq.append(openings[k]); mc.append(0); done.append(False); outcome.append(None)
    N = len(states)

    # Per-game material trace (A's perspective): peak A-edge, peak B-edge, final margin.
    peakA = [-99.0] * N; peakA_fen = [None] * N; peakA_ply = [0] * N
    peakB = [-99.0] * N; peakB_fen = [None] * N; peakB_ply = [0] * N
    final_margin = [0.0] * N

    print(f"Match: A=step{stepA} ({os.path.basename(args.checkpoint_a)})  vs  "
          f"B=step{stepB} ({os.path.basename(args.checkpoint_b)})")
    print(f"  {N} games ({n_pairs} paired openings x2 colors), {args.sims} sims, "
          f"opening {args.opening_plies} plies, cap {args.max_plies}, device {dev}\n")

    def net_to_move(i):
        white_to_move = states[i].current_player == 1
        return white_net[i] if white_to_move else ("B" if white_net[i] == "A" else "A")

    def record_material(i):
        mw = material_white(states[i].board)
        marginA = mw if white_net[i] == "A" else -mw  # >0 => A ahead
        final_margin[i] = marginA
        if marginA > peakA[i]:
            peakA[i] = marginA; peakA_fen[i] = states[i].board.fen(); peakA_ply[i] = mc[i]
        if -marginA > peakB[i]:
            peakB[i] = -marginA; peakB_fen[i] = states[i].board.fen(); peakB_ply[i] = mc[i]

    active = list(range(N))
    op = args.opening_plies
    ply = 0
    while active:
        single = {i: game.to_tensor(states[i]) for i in active}
        legal = {i: game.legal_actions(states[i]) for i in active}
        groups = {"A": [], "B": []}
        for i in active:
            if mc[i] >= op:
                groups[net_to_move(i)].append(i)
        roots = {}
        for tag, idxs in groups.items():
            if not idxs:
                continue
            obs_list = [stack_with_history(single[i], obs_hist[i], n_frames) for i in idxs]
            r = mcts[tag].run_batch(obs_list, [legal[i] for i in idxs], add_noise=False)
            for j, i in enumerate(idxs):
                roots[i] = r[j]

        still = []
        for i in active:
            if mc[i] < op:
                action = opening_seq[i][mc[i]] if mc[i] < len(opening_seq[i]) else random.choice(legal[i])
            else:
                action, _ = select_action(roots[i], temperature=0.0)
            obs_hist[i].append(single[i])
            state, _, _ = game.step(states[i], action)
            states[i] = state; mc[i] += 1
            record_material(i)
            if state.done:
                outcome[i] = state.winner; done[i] = True
            elif mc[i] >= args.max_plies:
                outcome[i] = 0.0; done[i] = True
            else:
                still.append(i)
        active = still
        ply += 1
        if ply % 10 == 0:
            print(f"  ply {ply}: {len(active)}/{N} games still active", flush=True)

    # ---- standard score from A's perspective ----
    aw = ad = al = 0
    a_white_w = a_white_dec = a_black_w = a_black_dec = 0
    scores = []
    for i in range(N):
        o = outcome[i]
        a_is_white = white_net[i] == "A"
        a_result = (o if a_is_white else -o)
        if a_result > 0: aw += 1; s = 1.0
        elif a_result < 0: al += 1; s = 0.0
        else: ad += 1; s = 0.5
        scores.append(s)
        if a_is_white:
            a_white_dec += (o != 0); a_white_w += (o > 0)
        else:
            a_black_dec += (o != 0); a_black_w += (o < 0)
    scores = np.array(scores); score = scores.mean()
    se = scores.std(ddof=1) / math.sqrt(N)
    lo, hi = score - 1.96 * se, score + 1.96 * se

    def elo(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return -400 * math.log10(1 / p - 1)

    print(f"\n{'='*66}\nRESULT (A = step {stepA}, relative to B = step {stepB})\n{'='*66}")
    print(f"  A: {aw} W  /  {ad} D  /  {al} L   (out of {N})")
    print(f"  draw rate: {ad/N:.1%}   decisive: {(aw+al)/N:.1%}")
    print(f"  score: {score:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"  Elo(A - B): {elo(score):+.0f}  (95% CI {elo(lo):+.0f} .. {elo(hi):+.0f})")
    print(f"  color sanity — A as white: {a_white_w}/{a_white_dec} decisive won; "
          f"A as black: {a_black_w}/{a_black_dec} decisive won")
    verdict = ("A clearly stronger" if lo > 0.5 else
               "B clearly stronger" if hi < 0.5 else
               "no significant difference (CI spans 50%)")
    print(f"  >>> {verdict}")

    # ---- material / conversion analysis ----
    a_res = [(outcome[i] if white_net[i] == "A" else -outcome[i]) for i in range(N)]
    draws = [i for i in range(N) if a_res[i] == 0]

    def count_edge(idxs, peak, thr):
        return sum(1 for i in idxs if peak[i] >= thr)

    print(f"\n{'='*66}\nMATERIAL / CONVERSION ANALYSIS (margin in pawns, A-perspective)\n{'='*66}")
    print(f"  drawn games: {len(draws)}/{N}")
    print(f"  among draws, side reached a PEAK material edge of >= T at some point:")
    print(f"     T=+2:  A-ahead {count_edge(draws,peakA,2):3d}   B-ahead {count_edge(draws,peakB,2):3d}")
    print(f"     T=+3:  A-ahead {count_edge(draws,peakA,3):3d}   B-ahead {count_edge(draws,peakB,3):3d}")
    print(f"     T=+5:  A-ahead {count_edge(draws,peakA,5):3d}   B-ahead {count_edge(draws,peakB,5):3d}")
    # final on-board margin at the drawn terminal position (sustained, not a spike)
    fa2 = sum(1 for i in draws if final_margin[i] >= 2)
    fb2 = sum(1 for i in draws if final_margin[i] <= -2)
    fa3 = sum(1 for i in draws if final_margin[i] >= 3)
    fb3 = sum(1 for i in draws if final_margin[i] <= -3)
    fa5 = sum(1 for i in draws if final_margin[i] >= 5)
    fb5 = sum(1 for i in draws if final_margin[i] <= -5)
    print(f"  among draws, material STILL ON THE BOARD at the drawn position (sustained edge):")
    print(f"     >=+2:  A-ahead {fa2:3d}   B-ahead {fb2:3d}")
    print(f"     >=+3:  A-ahead {fa3:3d}   B-ahead {fb3:3d}")
    print(f"     >=+5:  A-ahead {fa5:3d}   B-ahead {fb5:3d}")

    def conv(peak, who_perspective):
        # games where THIS side reached >=+3 at peak, and what happened (from that side's view)
        won = drew = lost = total = 0
        for i in range(N):
            edge = peak[i]
            if edge < 3:
                continue
            total += 1
            r = a_res[i] if who_perspective == "A" else -a_res[i]
            if r > 0: won += 1
            elif r == 0: drew += 1
            else: lost += 1
        return total, won, drew, lost

    tA, wA, dA, lA = conv(peakA, "A")
    tB, wB, dB, lB = conv(peakB, "B")
    print(f"\n  CONVERSION once a side reaches >= +3 material (peak), from that side's view:")
    print(f"     A reached +3 in {tA:3d} games -> {wA} won / {dA} drew / {lA} lost"
          f"   (convert {wA/max(1,tA):.0%})")
    print(f"     B reached +3 in {tB:3d} games -> {wB} won / {dB} drew / {lB} lost"
          f"   (convert {wB/max(1,tB):.0%})")

    # ---- dump the egregious unconverted draws ----
    notable = []
    for i in draws:
        edge = max(peakA[i], peakB[i])
        if edge >= 3:
            who = "A" if peakA[i] >= peakB[i] else "B"
            notable.append((edge, who, i))
    notable.sort(reverse=True)
    print(f"\n  unconverted draws with peak edge >= +3: {len(notable)} "
          f"(showing up to 12 largest)")
    for edge, who, i in notable[:12]:
        fen = peakA_fen[i] if who == "A" else peakB_fen[i]
        pk = peakA_ply[i] if who == "A" else peakB_ply[i]
        net = ("A=step%s" % stepA) if who == "A" else ("B=step%s" % stepB)
        print(f"     game {i:3d}: {net} reached +{int(edge)} at ply {pk}, final on-board "
              f"{final_margin[i]:+.0f} (A-persp), drew | {fen}")

    if args.dump:
        with open(args.dump, "w") as f:
            f.write(f"A=step{stepA} ({args.checkpoint_a})\nB=step{stepB} ({args.checkpoint_b})\n")
            f.write("game,white_net,a_result,peakA,peakA_ply,peakB,peakB_ply,final_margin_A,peak_fen\n")
            for i in range(N):
                who = "A" if peakA[i] >= peakB[i] else "B"
                fen = peakA_fen[i] if who == "A" else peakB_fen[i]
                f.write(f"{i},{white_net[i]},{a_res[i]:+.0f},{peakA[i]:.0f},{peakA_ply[i]},"
                        f"{peakB[i]:.0f},{peakB_ply[i]},{final_margin[i]:+.0f},{fen}\n")
        print(f"\n  per-game material table -> {args.dump}")


if __name__ == "__main__":
    main()
