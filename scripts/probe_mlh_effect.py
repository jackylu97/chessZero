"""Does the moves-left (MLH) search utility bias the search? The MLH head has
collapsed to ~7 plies everywhere (games are short draws). moves_left_mcts=True
adds sign(-Q)*min(slope*M, max_effect) to the PUCT score for |Q|>threshold,
nudging toward faster wins / slower losses.

Test on real positions: run numpy MCTS with MLH ON vs OFF; report
 - whether the picked move changes
 - what plies-to-end the MLH head predicts (collapsed?)
 - the value at root
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, chess
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer
from src.mcts.mcts import MCTS
from src.model.utils import support_to_scalar
torch.serialization.add_safe_globals([MuZeroConfig])
from scripts.eval_checkpoint_health import build_network

CKPT = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
BUF = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"


@torch.no_grad()
def main():
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = "cpu"
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = build_network(ckpt, game, cfg, "cpu")
    rb = ReplayBuffer(cfg); rb.load(BUF, game=game)
    games = [g for g in rb.buffer if not g.external_values and len(g) >= 8]
    rng = np.random.default_rng(7)

    # MLH head plies-to-end distribution across real positions.
    mls = []
    sample = rng.choice(len(games), size=200, replace=False)
    for gi in sample:
        gh = games[int(gi)]
        si = int(rng.integers(2, len(gh) - 1))
        obs = gh._stack_history(si, cfg.history_frames)
        h = net.representation(obs.unsqueeze(0))
        ml = net.predict_moves_left(h)
        mls.append(float(support_to_scalar(ml, getattr(net, "moves_left_support_size", 10)).item()))
    mls = np.array(mls)
    print(f"MLH head plies-to-end over 200 real positions: "
          f"mean={mls.mean():.1f} std={mls.std():.1f} min={mls.min():.1f} max={mls.max():.1f}")
    print(f"  (true plies-to-end varies 0..hundreds; collapse to a tiny constant = head is uninformative)")

    # Compare MCTS pick MLH-on vs MLH-off on positions with |root V|>0.3 (where MLH activates).
    cfg_on = get_config("chess_small"); cfg_on.device = "cpu"; cfg_on.num_simulations = 200
    cfg_off = get_config("chess_small"); cfg_off.device = "cpu"; cfg_off.num_simulations = 200
    cfg_off.moves_left_mcts = False
    mcts_on = MCTS(net, game, cfg_on, "cpu")
    mcts_off = MCTS(net, game, cfg_off, "cpu")
    assert mcts_on._use_ml_utility and not mcts_off._use_ml_utility, (mcts_on._use_ml_utility, mcts_off._use_ml_utility)

    n_diff = 0; n = 0
    for gi in rng.choice(len(games), size=30, replace=False):
        gh = games[int(gi)]
        si = int(rng.integers(2, len(gh) - 1))
        obs = gh._stack_history(si, cfg.history_frames)
        board = chess.Board(); ok = True
        for a in gh.actions[:si]:
            mv = _action_to_move(int(a), board)
            if mv is None or mv not in board.legal_moves:
                ok = False; break
            board.push(mv)
        if not ok or board.is_game_over():
            continue
        legal = game.legal_actions(_state(game, board))
        r_on = mcts_on.run(obs, legal, add_noise=False)
        r_off = mcts_off.run(obs, legal, add_noise=False)
        a_on = int(r_on.child_actions[int(np.argmax(r_on.child_visits))])
        a_off = int(r_off.child_actions[int(np.argmax(r_off.child_visits))])
        n += 1
        if a_on != a_off:
            n_diff += 1
    print(f"\nMCTS pick changes between MLH-on and MLH-off: {n_diff}/{n} = {n_diff/max(n,1):.1%}")
    print(f"  (high => the collapsed MLH head is actively steering move choice)")


def _state(game, board):
    st = game.reset(); st.board = board.copy()
    st.current_player = 1 if board.turn == chess.WHITE else -1
    return st


if __name__ == "__main__":
    main()
