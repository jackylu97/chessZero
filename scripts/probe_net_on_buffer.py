"""Run the trained net's value head + MCTS on real buffer positions to test:
is the value head RIGHT but search still draws (BUG) or can't tell (fundamental)?
CPU only."""
import argparse, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.training.replay_buffer import GameHistory, stack_with_history
from src.model.utils import wdl_to_scalar
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
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--sims", type=int, default=200)
    args = ap.parse_args()

    cfg = get_config("chess_small")
    cfg.device = "cpu"
    HF = cfg.history_frames
    game = ChessGame()
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    net = build_network(ckpt, game, cfg, "cpu")
    print(f"loaded {args.ckpt} step={ckpt.get('step')}  HF={HF} draw_score={cfg.draw_score}")

    games = load_buffer_games(args.buf, game, args.limit)
    rep_games = [g for g in games if g.draw_by_repetition]
    dec_games = [g for g in games if g.game_outcome != 0.0]

    @torch.no_grad()
    def eval_obs(obs):
        h = net.representation(obs.unsqueeze(0))
        pl, vl = net.prediction(h)
        wdl = F.softmax(vl.float(), -1)[0].numpy()
        v = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).item()
        v0 = wdl_to_scalar(vl.float(), draw_score=0.0).item()
        return wdl, v, v0

    def trace_net(g, label, last_n=10):
        n = len(g)
        print(f"\n{'='*100}\n{label}: len={n} outcome={g.game_outcome} by_rep={g.draw_by_repetition}")
        print(f"  ply stm | net WDL[W,D,L]    netV(ds={cfg.draw_score})  netV(ds=0) | rootV(stored)")
        plies = sorted(set([0, n//4, n//2] + list(range(max(0, n - last_n), n - 1))))
        for ply in plies:
            obs = g._stack_history(ply, HF)
            wdl, v, v0 = eval_obs(obs)
            rv = g.root_values[ply] if ply < len(g.root_values) else float("nan")
            stm = "W" if ply % 2 == 0 else "B"
            print(f"  {ply:4d} {stm}  | [{wdl[0]:.3f},{wdl[1]:.3f},{wdl[2]:.3f}]   {v:+.3f}      {v0:+.3f}   | {rv:+.3f}")

    for i, g in enumerate(rep_games[:2]):
        trace_net(g, f"REP-DRAW #{i}")
    for i, g in enumerate(dec_games[:2]):
        trace_net(g, f"DECISIVE #{i} (outcome={g.game_outcome})")

    # ---- MCTS on a few rep-draw positions near the repetition: child Q spread? ----
    print(f"\n\n{'#'*100}\nMCTS on rep-draw positions (do children have distinct Q? does search pick the shuffle?)")
    from src.mcts.mcts import MCTS
    cfg.num_simulations = args.sims
    mcts = MCTS(net, game, cfg, "cpu")

    # Replay a rep-draw game to get board states + legal actions at chosen plies.
    g = rep_games[0]
    # Rebuild boards by replaying.
    state = game.reset()
    boards = [state]
    for a in g.actions:
        state, _, _ = game.step(state, a)
        boards.append(state)
    n = len(g)
    for ply in [n - 6, n - 4, n - 2]:
        if ply < 0:
            continue
        st = boards[ply]
        legal = game.legal_actions(st)
        obs = g._stack_history(ply, HF)
        root = mcts.run(obs, legal, add_noise=False)
        order = np.argsort(-root.child_visits)
        stm = "W" if ply % 2 == 0 else "B"
        print(f"\n  ply {ply} ({stm} to move) root.value={root.value:+.3f} legal={len(legal)}  "
              f"actual_played_action={g.actions[ply] if ply < len(g.actions) else None}")
        played = g.actions[ply] if ply < len(g.actions) else None
        for j in order[:6]:
            a = int(root.child_actions[j]); vis = float(root.child_visits[j])
            q = (root.child_value_sums[j] / vis) if vis > 0 else 0.0
            pr = float(root.child_priors[j])
            mv = _action_to_move(a, st.board)
            san = st.board.san(mv) if (mv and mv in st.board.legal_moves) else f"a{a}"
            tag = " <== PLAYED" if a == played else ""
            print(f"      {san:7s} visits={vis:5.0f} Q={q:+.3f} prior={pr:.3f}{tag}")


if __name__ == "__main__":
    main()
