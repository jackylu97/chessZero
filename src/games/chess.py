"""Chess game implementation using python-chess."""

import chess
import numpy as np
import torch

from .base import Game, GameState


# Action encoding: 64 source squares × 73 move types = 4672 actions
# Move types: 56 queen moves (7 distances × 8 directions) + 8 knight moves + 9 underpromotions
NUM_MOVE_TYPES = 73
ACTION_SPACE = 64 * NUM_MOVE_TYPES  # 4672


def _move_to_action(move: chess.Move) -> int:
    """Encode a chess move as a flat action index."""
    from_sq = move.from_square
    to_sq = move.to_square

    from_row, from_col = divmod(from_sq, 8)
    to_row, to_col = divmod(to_sq, 8)
    dr = to_row - from_row
    dc = to_col - from_col

    # Knight moves
    knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                    (1, -2), (1, 2), (2, -1), (2, 1)]
    if (dr, dc) in knight_moves:
        move_type = 56 + knight_moves.index((dr, dc))
    # Underpromotions
    elif move.promotion and move.promotion != chess.QUEEN:
        promo_map = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}
        direction = 0 if dc == 0 else (1 if dc == 1 else 2)
        move_type = 64 + promo_map[move.promotion] * 3 + direction
    else:
        # Queen-type moves (including queen promotions)
        # 8 directions × 7 distances
        if dr == 0:
            direction = 0 if dc > 0 else 4
            distance = abs(dc) - 1
        elif dc == 0:
            direction = 2 if dr > 0 else 6
            distance = abs(dr) - 1
        elif dr > 0 and dc > 0:
            direction = 1
            distance = abs(dr) - 1
        elif dr > 0 and dc < 0:
            direction = 3
            distance = abs(dr) - 1
        elif dr < 0 and dc > 0:
            direction = 7
            distance = abs(dr) - 1
        else:  # dr < 0 and dc < 0
            direction = 5
            distance = abs(dr) - 1
        move_type = direction * 7 + distance

    return from_sq * NUM_MOVE_TYPES + move_type


def _action_to_move(action: int, board: chess.Board) -> chess.Move | None:
    """Decode flat action index back to a chess move, validating against board."""
    from_sq = action // NUM_MOVE_TYPES
    move_type = action % NUM_MOVE_TYPES

    from_row, from_col = divmod(from_sq, 8)

    if move_type < 56:
        # Queen-type move
        direction = move_type // 7
        distance = move_type % 7 + 1
        dir_map = {
            0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1),
            4: (0, -1), 5: (-1, -1), 6: (-1, 0), 7: (-1, 1),
        }
        dr, dc = dir_map[direction]
        to_row = from_row + dr * distance
        to_col = from_col + dc * distance
    elif move_type < 64:
        # Knight move
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2), (1, 2), (2, -1), (2, 1)]
        dr, dc = knight_moves[move_type - 56]
        to_row = from_row + dr
        to_col = from_col + dc
    else:
        # Underpromotion
        idx = move_type - 64
        promo_idx = idx // 3  # 0=rook, 1=bishop, 2=knight
        direction = idx % 3  # 0=straight, 1=right capture, 2=left capture
        promo_map = {0: chess.ROOK, 1: chess.BISHOP, 2: chess.KNIGHT}
        dc = [0, 1, -1][direction]
        dr = 1 if board.turn == chess.WHITE else -1
        to_row = from_row + dr
        to_col = from_col + dc

        if 0 <= to_row < 8 and 0 <= to_col < 8:
            to_sq = to_row * 8 + to_col
            return chess.Move(from_sq, to_sq, promotion=promo_map[promo_idx])
        return None

    if 0 <= to_row < 8 and 0 <= to_col < 8:
        to_sq = to_row * 8 + to_col
        # Check if this is a queen promotion
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            if (board.turn == chess.WHITE and to_row == 7) or \
               (board.turn == chess.BLACK and to_row == 0):
                return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        return chess.Move(from_sq, to_sq)
    return None


class ChessGame(Game):
    board_size = (8, 8)
    action_space_size = ACTION_SPACE  # 4672
    num_planes = 19  # 6 piece types × 2 colors + castling (4) + en passant (1) + turn (1) + move count (1)

    def reset(self) -> GameState:
        return GameState(
            board=chess.Board(),
            current_player=1,  # white = 1, black = -1
        )

    def step(self, state: GameState, action: int) -> tuple[GameState, float, bool]:
        board = state.board.copy()
        move = _action_to_move(action, board)
        assert move is not None and move in board.legal_moves, f"Illegal move: action {action}"
        board.push(move)

        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                winner = 1
                reward = 1.0 if state.current_player == 1 else -1.0
            elif result == "0-1":
                winner = -1
                reward = 1.0 if state.current_player == -1 else -1.0
            else:
                winner = 0
                reward = 0.0
            return GameState(board=board, current_player=-state.current_player,
                             done=True, winner=winner), reward, True

        return GameState(board=board, current_player=-state.current_player), 0.0, False

    def legal_actions(self, state: GameState) -> list[int]:
        return [_move_to_action(m) for m in state.board.legal_moves]

    def to_tensor(self, state: GameState) -> torch.Tensor:
        board = state.board
        planes = np.zeros((self.num_planes, 8, 8), dtype=np.float32)

        # Piece planes: P=0, N=1, B=2, R=3, Q=4, K=5 for white; +6 for black.
        # Always encoded from white's perspective (row 0 = rank 1).
        # The network learns separate white/black strategies via the turn plane.
        # Proper perspective flipping (like AlphaZero) would also require flipping
        # action indices in legal_actions() — deferred as a future improvement.
        piece_map = board.piece_map()
        for sq, piece in piece_map.items():
            row, col = divmod(sq, 8)
            plane_idx = piece.piece_type - 1  # 0-5
            if piece.color == chess.BLACK:
                plane_idx += 6
            planes[plane_idx, row, col] = 1.0

        # Castling rights
        planes[12, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
        planes[13, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
        planes[14, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
        planes[15, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

        # En passant
        if board.ep_square is not None:
            row, col = divmod(board.ep_square, 8)
            planes[16, row, col] = 1.0

        # Current player
        planes[17, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

        # Move count (normalized)
        planes[18, :, :] = min(board.fullmove_number / 200.0, 1.0)

        return torch.from_numpy(planes)

    def render(self, state: GameState) -> str:
        return str(state.board)

    def clone_state(self, state: GameState) -> GameState:
        return GameState(
            board=state.board.copy(),
            current_player=state.current_player,
            done=state.done,
            winner=state.winner,
        )
