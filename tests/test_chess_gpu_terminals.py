"""Cross-validate explicit terminal detection (mate / stalemate / 50-move /
threefold / ply-cap) between ChessGame and GpuChessGame.

`test_step_batch_random_games` exercises terminals indirectly via random
play, but a random-game corpus is unreliable for ply-cap (almost no game
runs 400 plies) and for fast 50-move / threefold detection. These targeted
positions force each terminal to fire exactly once and assert the GPU's
done/winner/reward triple matches python-chess.

Insufficient material is intentionally NOT tested — GpuChessGame doesn't
model it (see comment in test_step_batch_random_games).
"""
import chess
import torch

from src.games.chess import ChessGame, _move_to_action
from src.games.chess_gpu import GpuChessGame


def _step_both_and_compare(fen: str, move_uci: str, expected_done: bool,
                            expected_reward: float, expected_winner_state: int) -> None:
    """Build state from FEN, step both engines with `move_uci`, compare results.

    `expected_winner_state` is the value of GameState.winner (and equivalently
    GpuChessGame's winner field) AFTER the step: +1 white, -1 black, 0 draw/none.
    """
    cg = ChessGame()
    gg = GpuChessGame()

    # Build CPU side.
    board = chess.Board(fen)
    from src.games.base import GameState
    cpu_state = GameState(
        board=board.copy(),
        current_player=1 if board.turn else -1,
    )

    # Build GPU side.
    gpu_state = gg.from_python_chess([board.copy()])

    # Decode the move via python-chess (handles SAN/UCI), encode to action.
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        # Some test moves include promotion suffix — try parse via SAN fallback.
        move = board.parse_san(move_uci)
    assert move in board.legal_moves, f"test move {move_uci} not legal in {fen}"
    action_stm = _move_to_action(move, board.turn)

    cpu_after, cpu_reward, cpu_done = cg.step(cpu_state, action_stm)
    actions_t = torch.tensor([action_stm], dtype=torch.int64)
    gpu_after, gpu_reward_t, gpu_done_t = gg.step_batch(gpu_state, actions_t)

    gpu_reward = float(gpu_reward_t[0].item())
    gpu_done = bool(gpu_done_t[0].item())
    gpu_winner = int(gpu_after.winner[0].item())

    # Assert oracle agrees with our expectation (sanity — protects against typos).
    assert cpu_done == expected_done, f"oracle done={cpu_done}, expected {expected_done}"
    assert cpu_reward == expected_reward, f"oracle reward={cpu_reward}, expected {expected_reward}"
    assert cpu_after.winner == expected_winner_state, (
        f"oracle winner={cpu_after.winner}, expected {expected_winner_state}"
    )

    # Assert GPU matches oracle.
    assert gpu_done == cpu_done, f"GPU done={gpu_done} vs CPU {cpu_done}"
    assert gpu_reward == cpu_reward, f"GPU reward={gpu_reward} vs CPU {cpu_reward}"
    assert gpu_winner == cpu_after.winner, f"GPU winner={gpu_winner} vs CPU {cpu_after.winner}"


def test_terminal_mate_back_rank_white():
    """Back-rank mate by white. White plays Re8#: black king on g8 has no escape."""
    # White rook e1 → e8 mate. Black king g8, black pawns f7/g7/h7.
    fen = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"
    _step_both_and_compare(fen, "e1e8", expected_done=True,
                           expected_reward=1.0, expected_winner_state=1)


def test_terminal_mate_back_rank_black():
    """Mirror: black plays Re1# — flips reward & winner sign."""
    fen = "4r2k/8/8/8/8/8/5PPP/6K1 b - - 0 1"
    _step_both_and_compare(fen, "e8e1", expected_done=True,
                           expected_reward=1.0, expected_winner_state=-1)


def test_terminal_stalemate():
    """White plays Qb6 → black king on a8 has no legal moves and isn't in check."""
    # Pre: white Q on c6. Move Qb6 stalemates kings on a8/h1.
    fen = "k7/8/2Q5/8/8/8/8/7K w - - 0 1"
    _step_both_and_compare(fen, "c6b6", expected_done=True,
                           expected_reward=0.0, expected_winner_state=0)


def test_terminal_seventy_five_move():
    """Halfmove clock at 149; a non-capture non-pawn move pushes to 150 → auto draw.

    python-chess auto-terminates on the 75-move rule (halfmove_clock >= 150),
    not the claimable 50-move rule. ChessGame uses `is_game_over(claim_draw=False)`
    so picks up only the auto rule; GpuChessGame matches.
    """
    fen = "4k3/8/8/8/8/8/8/N3K2N w - - 149 75"
    _step_both_and_compare(fen, "a1c2", expected_done=True,
                           expected_reward=0.0, expected_winner_state=0)


def test_terminal_threefold_repetition():
    """Rook shuffle on an otherwise quiet board reaches threefold at ply 8.

    K+N vs K (and K+B vs K, KNN vs K) are insufficient material → python-chess
    terminates immediately. Use K+R vs K+R with no castling rights so each
    side can shuffle a rook reversibly. Sequence is 8 plies: position appears
    at ply 0, 4, and 8 → threefold fires.
    """
    fen = "3k3r/8/8/8/8/8/8/R3K3 w - - 0 1"
    cg = ChessGame()
    gg = GpuChessGame()

    moves = ["a1a2", "h8h7", "a2a1", "h7h8",
             "a1a2", "h8h7", "a2a1", "h7h8"]

    from src.games.base import GameState
    board = chess.Board(fen)
    cpu_state = GameState(
        board=board.copy(),
        current_player=1 if board.turn else -1,
    )
    gpu_state = gg.from_python_chess([board.copy()])

    cpu_done = False
    gpu_done = False
    cpu_reward = 0.0
    gpu_reward = 0.0
    for ply, uci in enumerate(moves):
        move = chess.Move.from_uci(uci)
        action_stm = _move_to_action(move, cpu_state.board.turn)

        cpu_state, cpu_reward, cpu_done = cg.step(cpu_state, action_stm)
        a_t = torch.tensor([action_stm], dtype=torch.int64)
        gpu_state, gpu_reward_t, gpu_done_t = gg.step_batch(gpu_state, a_t)
        gpu_reward = float(gpu_reward_t[0].item())
        gpu_done = bool(gpu_done_t[0].item())

        # Until the final ply, neither side should have terminated.
        if ply < len(moves) - 1:
            assert not cpu_done, f"CPU terminated early at ply {ply}"
            assert not gpu_done, f"GPU terminated early at ply {ply}"

    # Final ply: threefold repetition.
    assert cpu_done, "oracle should detect threefold at ply 7 (0-indexed)"
    assert gpu_done == cpu_done, f"GPU done={gpu_done} vs CPU {cpu_done}"
    assert gpu_reward == cpu_reward == 0.0, (
        f"reward CPU={cpu_reward} GPU={gpu_reward}"
    )
    assert int(gpu_state.winner[0].item()) == cpu_state.winner == 0


def test_terminal_ply_cap():
    """Construct a position with ply()==399; one move pushes to 400 → cap.

    python-chess derives ply() from fullmove_number + turn:
        ply = 2*(fullmove - 1) + (turn==BLACK)
    So fullmove=200, black to move → ply=399. After any move, ply=400=max_plies.
    """
    # Two kings + two knights on quiet ranks so the move is legal.
    fen = "4k3/8/8/8/8/8/8/N3K2N b - - 0 200"
    board = chess.Board(fen)
    assert board.ply() == 399, f"FEN setup wrong: ply={board.ply()}"

    _step_both_and_compare(fen, "e8d8", expected_done=True,
                           expected_reward=0.0, expected_winner_state=0)
