"""Decisive experiment: at a real rep position, does giving MCTS repetition
awareness (forced_draw_actions, present in numpy BatchedMCTS but ABSENT from the
live tensor engine) change which move the search picks? If yes => search-side
fixable bug, not 'value head can't tell'. CPU."""
import argparse, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import GameHistory
from scripts.eval_checkpoint_health import build_network


def load_buffer_games(path, game, limit=None):
    games = []
    with open(path, "rb") as f:
        header = pickle.load(f)
        version = header["version"]
        for i in range(header["n_records"]):
            record, priority = pickle.load(f)
            if version == 3:
                record = GameHistory.from_compact_dict(record, game)
            games.append(record)
            if limit and len(games) >= limit:
                break
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt")
    ap.add_argument("--buf", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--sims", type=int, default=200)
    args = ap.parse_args()
    cfg = get_config("chess_small"); cfg.device = "cpu"; cfg.num_simulations = args.sims
    HF = cfg.history_frames
    game = ChessGame()
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    net = build_network(ckpt, game, cfg, "cpu")
    games = load_buffer_games(args.buf, game, args.limit)
    rep = [g for g in games if g.draw_by_repetition]

    from src.mcts.mcts import BatchedMCTS
    mcts = BatchedMCTS(net, game, cfg, "cpu")

    # Find a rep game where a shuffle-into-repetition move exists at a late ply.
    examined = 0
    for g in rep:
        if len(g.actions) < 10:
            continue
        # Replay to build boards.
        st = game.reset(); boards = [st]
        for a in g.actions:
            st, _, _ = game.step(st, a); boards.append(st)
        n = len(g.actions)
        # At a late ply, find the action(s) that, if played, create a 3-fold repetition.
        for ply in range(n - 4, n):
            if ply < 0:
                continue
            st = boards[ply]
            b = st.board
            legal = game.legal_actions(st)
            # Which legal actions lead to is_repetition(3)?
            draw_actions = []
            for a in legal:
                mv = _action_to_move(int(a), b)
                if mv is None or mv not in b.legal_moves:
                    continue
                nb = b.copy(); nb.push(mv)
                if nb.is_repetition(3):
                    draw_actions.append(int(a))
            if not draw_actions:
                continue
            obs = g._stack_history(ply, HF)
            stm = "W" if ply % 2 == 0 else "B"
            print(f"\n{'='*80}\nGAME len={n} ply={ply} ({stm} to move)  rep-creating actions={len(draw_actions)}")
            played = g.actions[ply] if ply < len(g.actions) else None

            def show(root, label):
                order = np.argsort(-root.child_visits)
                print(f"  [{label}] root.value={root.value:+.3f}")
                for j in order[:5]:
                    a = int(root.child_actions[j]); vis = float(root.child_visits[j])
                    q = (root.child_value_sums[j] / vis) if vis > 0 else 0.0
                    mv = _action_to_move(a, b)
                    san = b.san(mv) if (mv and mv in b.legal_moves) else f"a{a}"
                    isdraw = " (REP-DRAW move)" if a in draw_actions else ""
                    tag = " <==PLAYED" if a == played else ""
                    print(f"      {san:7s} visits={vis:5.0f} Q={q:+.3f}{isdraw}{tag}")
                top = int(root.child_actions[order[0]])
                return top in draw_actions

            # WITHOUT repetition awareness (== the live tensor engine).
            root_no = mcts.run_batch([obs], [legal], add_noise=False, forced_draw_actions=None)[0]
            picks_draw_no = show(root_no, "NO rep-awareness (live tensor engine)")
            # WITH repetition awareness (numpy oracle's forced_draw).
            root_yes = mcts.run_batch([obs], [legal], add_noise=False,
                                      forced_draw_actions=[set(draw_actions)])[0]
            picks_draw_yes = show(root_yes, "WITH rep-awareness (forced_draw)")
            print(f"  => top move is a REP-DRAW move?  NO-aware: {picks_draw_no}   WITH-aware: {picks_draw_yes}")
            examined += 1
            break
        if examined >= 4:
            break


if __name__ == "__main__":
    main()
