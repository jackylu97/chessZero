"""Absolute-strength benchmark: the model (MCTS, no TB probe) vs a strength-limited
Stockfish. Puts the model on a real Elo axis instead of relative self-play Elo.

Stockfish UCI_Elo floor is 1320; for weaker, use --sf-skill (Skill Level 0-20).
Paired games (colors swapped) so opening/color luck cancels.

Run: .venv/bin/python scripts/eval_vs_stockfish.py --checkpoint <ckpt.pt> \
        --game chess_small --sf-elo 1320 --games 30 --sims 160
"""
import argparse, math, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess, chess.engine

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.mcts.mcts import BatchedMCTS, select_action
from src.training.replay_buffer import stack_with_history
from scripts.eval_checkpoint_health import build_network


def model_move(mcts, game, board, frames, HF, cur_obs):
    obs = stack_with_history(cur_obs, frames, HF)
    state = game.reset_from_fen(board.fen())
    legal = game.legal_actions(state)
    root = mcts.run_batch([obs], [legal], add_noise=False)[0]
    from src.games.chess import _action_to_move
    a, _ = select_action(root, temperature=0.0)
    acts = root.child_actions
    action = int(acts[int(np.argmax(root.child_visits))])
    return _action_to_move(action, board)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--game", default="chess_small")
    ap.add_argument("--sf-elo", type=int, default=1320, help="UCI_Elo (>=1320), calibrated.")
    ap.add_argument("--sf-skill", type=int, default=None, help="Skill Level 0-20 (weaker than 1320).")
    ap.add_argument("--sf-movetime", type=float, default=0.1, help="Stockfish seconds/move.")
    ap.add_argument("--stockfish", default="tools/stockfish/stockfish")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    dev = args.device
    game = ChessGame()
    cfg = get_config(args.game); cfg.device = dev; cfg.num_simulations = args.sims
    HF = getattr(cfg, "history_frames", 1)
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev); net.eval()
    mcts = BatchedMCTS(net, game, cfg, dev)

    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    if args.sf_skill is not None:
        eng.configure({"Skill Level": args.sf_skill})
        opp = f"Stockfish Skill {args.sf_skill}"
    else:
        eng.configure({"UCI_LimitStrength": True, "UCI_Elo": args.sf_elo})
        opp = f"Stockfish UCI_Elo {args.sf_elo}"
    limit = chess.engine.Limit(time=args.sf_movetime)

    print(f"model step {ckpt.get('step','?')}  vs  {opp}  | {args.games} games, {args.sims} sims/move\n")
    W = D = L = 0
    n_pairs = args.games // 2
    rng = random.Random(args.seed)
    for k in range(n_pairs):
        # random opening (shared by the pair)
        opening = []
        b = chess.Board()
        for _ in range(args.opening_plies):
            mv = rng.choice(list(b.legal_moves)); opening.append(mv); b.push(mv)
        for model_is_white in (True, False):
            board = chess.Board()
            for mv in opening:
                board.push(mv)
            frames = []
            while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
                cur_obs = game.to_tensor(game.reset_from_fen(board.fen()))
                model_to_move = (board.turn == chess.WHITE) == model_is_white
                if model_to_move:
                    mv = model_move(mcts, game, board, frames, HF, cur_obs)
                    if mv is None or mv not in board.legal_moves:
                        mv = next(iter(board.legal_moves))
                else:
                    mv = eng.play(board, limit).move
                frames.append(cur_obs)
                board.push(mv)
            # score from the model's POV
            res = board.result(claim_draw=True)
            if res == "1/2-1/2" or board.ply() >= args.max_plies and res == "*":
                D += 1
            elif (res == "1-0") == model_is_white:
                W += 1
            else:
                L += 1
        done = (k + 1) * 2
        print(f"  after {done:3d} games: {W}W {D}D {L}L", flush=True)
    eng.quit()

    N = W + D + L
    score = (W + 0.5 * D) / max(1, N)
    se = math.sqrt(max(score * (1 - score), 1e-9) / max(1, N))
    lo, hi = score - 1.96 * se, score + 1.96 * se
    def elo(p): p = min(max(p, 1e-6), 1 - 1e-6); return -400 * math.log10(1 / p - 1)
    print(f"\n{'='*56}\nRESULT vs {opp}\n{'='*56}")
    print(f"  model: {W}W / {D}D / {L}L  (out of {N})")
    print(f"  score: {score:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"  Elo(model - opponent): {elo(score):+.0f}  (95% CI {elo(lo):+.0f} .. {elo(hi):+.0f})")
    print(f"  => model abs Elo ≈ {args.sf_elo if args.sf_skill is None else '?'} {elo(score):+.0f}"
          if args.sf_skill is None else f"  (Skill {args.sf_skill} is uncalibrated; ~1100-1350)")


if __name__ == "__main__":
    main()
