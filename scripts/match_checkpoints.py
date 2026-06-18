"""Head-to-head match between two checkpoints — the cleanest progress signal
(beats-past-self, AlphaZero's own yardstick). External outcome, no Stockfish
yardstick / self-target confound.

Plays paired games: G random openings, each played twice with colors swapped
(A-white/B-white) so opening luck and color bias cancel. After the random
opening both sides play their deterministic MCTS best move (no Dirichlet noise).

Run:
  .venv/bin/python scripts/match_checkpoints.py \
      --checkpoint-a checkpoints/chess/<run>/checkpoint_61000.pt \
      --checkpoint-b checkpoints/chess/<run>/checkpoint_40000.pt \
      --games 400 --sims 200 --device cuda
"""
import argparse, math, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.mcts.mcts import BatchedMCTS, select_action
from src.training.replay_buffer import stack_with_history
from scripts.eval_checkpoint_health import build_network


def load_net(path, game, cfg, dev):
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    return build_network(ckpt, game, cfg, dev), ckpt.get("step", "?")


def gen_openings(game, n_openings, plies, seed):
    """n_openings random legal opening lines (lists of action ints), length `plies`."""
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
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=300, help="declare draw past this (draw-basin shuffles)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = args.device
    game = ChessGame()
    cfg = get_config("chess"); cfg.device = dev
    cfg.num_simulations = args.sims
    n_frames = getattr(cfg, "history_frames", 1)
    asz = game.action_space_size

    netA, stepA = load_net(args.checkpoint_a, game, cfg, dev)
    netB, stepB = load_net(args.checkpoint_b, game, cfg, dev)
    mcts = {"A": BatchedMCTS(netA, game, cfg, dev), "B": BatchedMCTS(netB, game, cfg, dev)}

    n_pairs = args.games // 2
    openings = gen_openings(game, n_pairs, args.opening_plies, args.seed)

    # Build the game table. Two games per opening: white_net A then B.
    states, obs_hist, white_net, opening_seq, mc, done, outcome = [], [], [], [], [], [], []
    for k in range(n_pairs):
        for wn in ("A", "B"):
            states.append(game.reset()); obs_hist.append([]); white_net.append(wn)
            opening_seq.append(openings[k]); mc.append(0); done.append(False); outcome.append(None)
    N = len(states)
    print(f"Match: A=step{stepA} ({os.path.basename(args.checkpoint_a)})  vs  "
          f"B=step{stepB} ({os.path.basename(args.checkpoint_b)})")
    print(f"  {N} games ({n_pairs} paired openings x2 colors), {args.sims} sims, "
          f"opening {args.opening_plies} plies, cap {args.max_plies}, device {dev}\n")

    def net_to_move(i):
        white_to_move = states[i].current_player == 1
        return white_net[i] if white_to_move else ("B" if white_net[i] == "A" else "A")

    active = list(range(N))
    op = args.opening_plies
    ply = 0
    while active:
        single = {i: game.to_tensor(states[i]) for i in active}
        legal = {i: game.legal_actions(states[i]) for i in active}
        # MCTS phase games, grouped by the net to move.
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
            if state.done:
                outcome[i] = state.winner; done[i] = True
            elif mc[i] >= args.max_plies:
                outcome[i] = 0.0; done[i] = True  # length-cap shuffle => draw
            else:
                still.append(i)
        active = still
        ply += 1
        if ply % 10 == 0:
            print(f"  ply {ply}: {len(active)}/{N} games still active", flush=True)

    # Score from A's perspective.
    aw = ad = al = 0
    a_white_w = a_white_dec = a_black_w = a_black_dec = 0
    scores = []
    for i in range(N):
        o = outcome[i]
        a_is_white = white_net[i] == "A"
        a_result = (o if a_is_white else -o)  # +1 A wins, 0 draw, -1 A loses
        if a_result > 0: aw += 1; s = 1.0
        elif a_result < 0: al += 1; s = 0.0
        else: ad += 1; s = 0.5
        scores.append(s)
        if a_is_white:
            a_white_dec += (o != 0); a_white_w += (o > 0)
        else:
            a_black_dec += (o != 0); a_black_w += (o < 0)
    scores = np.array(scores)
    score = scores.mean()
    se = scores.std(ddof=1) / math.sqrt(N)
    lo, hi = score - 1.96 * se, score + 1.96 * se

    def elo(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return -400 * math.log10(1 / p - 1)

    print(f"\n{'='*64}\nRESULT (A = step {stepA}, relative to B = step {stepB})\n{'='*64}")
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


if __name__ == "__main__":
    main()
