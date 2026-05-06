"""Chess game implementation using python-chess."""

import chess
import numpy as np
import torch

from .base import Game, GameState


# Action encoding: 64 source squares × 73 move types = 4672 actions
# Move types: 56 queen moves (7 distances × 8 directions) + 8 knight moves + 9 underpromotions
NUM_MOVE_TYPES = 73
ACTION_SPACE = 64 * NUM_MOVE_TYPES  # 4672


def _move_to_action(move: chess.Move, turn: bool = chess.WHITE) -> int:
    """Encode a chess move as a flat action index in STM perspective.

    Actions are always encoded from the side-to-move's perspective (AlphaZero
    convention): the board is rank-flipped for black so that "forward" is always
    increasing rank. Pass ``turn`` so the flip is applied correctly.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    if turn == chess.BLACK:
        from_sq ^= 56
        to_sq ^= 56

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
    """Decode flat action index (STM perspective) back to a chess move.

    The action is encoded from the side-to-move's perspective. For black,
    from/to squares are un-flipped to absolute board coordinates.
    """
    from_sq = action // NUM_MOVE_TYPES
    move_type = action % NUM_MOVE_TYPES
    from_row, from_col = divmod(from_sq, 8)
    promotion = None

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
        # Underpromotion — always dr=+1 in STM space (pawn moves "north")
        idx = move_type - 64
        promo_idx = idx // 3
        direction = idx % 3
        promo_map = {0: chess.ROOK, 1: chess.BISHOP, 2: chess.KNIGHT}
        promotion = promo_map[promo_idx]
        dc = [0, 1, -1][direction]
        to_row = from_row + 1
        to_col = from_col + dc

    if not (0 <= to_row < 8 and 0 <= to_col < 8):
        return None

    to_sq = to_row * 8 + to_col

    # Un-flip from STM space to absolute board coordinates
    if board.turn == chess.BLACK:
        from_sq ^= 56
        to_sq ^= 56

    # Detect queen promotion (pawn reaching back rank)
    if promotion is None:
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            abs_to_rank = to_sq // 8
            if abs_to_rank == 0 or abs_to_rank == 7:
                promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


class ChessGame(Game):
    board_size = (8, 8)
    action_space_size = ACTION_SPACE  # 4672
    num_planes = 19  # 6 piece types × 2 colors + castling (4) + en passant (1) + turn (1) + move count (1)
    max_plies = 400  # force draw after 400 plies (200 fullmoves); 200 was too tight — untrained models can't resolve in that horizon, starves value head of real terminal labels

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

        hit_cap = board.ply() >= self.max_plies
        # claim_draw=False → only automatic terminations (mate, stalemate, insufficient,
        # fivefold, 75-move). We add strict threefold ourselves: is_repetition(3) means
        # the current position has *actually* occurred 3 times, not that a claim is one
        # ply away (which is what claim_draw=True would imply and caused premature draws).
        natural_end = board.is_game_over(claim_draw=False)
        strict_threefold = board.is_repetition(3)
        if natural_end or strict_threefold or hit_cap:
            if natural_end:
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
            else:
                winner = 0
                reward = 0.0
            return GameState(board=board, current_player=-state.current_player,
                             done=True, winner=winner), reward, True

        return GameState(board=board, current_player=-state.current_player), 0.0, False

    def legal_actions(self, state: GameState) -> list[int]:
        return [_move_to_action(m, state.board.turn) for m in state.board.legal_moves]

    def to_tensor(self, state: GameState) -> torch.Tensor:
        """Observation from the side-to-move's perspective (AlphaZero convention).

        Planes 0-5 = own pieces (P,N,B,R,Q,K), 6-11 = opponent pieces.
        When black to move the board is rank-flipped so own back rank is row 0.
        """
        board = state.board
        planes = np.zeros((self.num_planes, 8, 8), dtype=np.float32)
        is_black = board.turn == chess.BLACK

        piece_map = board.piece_map()
        for sq, piece in piece_map.items():
            row, col = divmod(sq, 8)
            if is_black:
                row = 7 - row
            plane_idx = piece.piece_type - 1  # 0-5
            is_own = (piece.color == chess.WHITE) != is_black
            if not is_own:
                plane_idx += 6
            planes[plane_idx, row, col] = 1.0

        # Castling rights — planes 12-13 = own, 14-15 = opponent
        own_color = chess.BLACK if is_black else chess.WHITE
        opp_color = chess.WHITE if is_black else chess.BLACK
        planes[12, :, :] = float(board.has_kingside_castling_rights(own_color))
        planes[13, :, :] = float(board.has_queenside_castling_rights(own_color))
        planes[14, :, :] = float(board.has_kingside_castling_rights(opp_color))
        planes[15, :, :] = float(board.has_queenside_castling_rights(opp_color))

        # En passant — flip square for black
        if board.ep_square is not None:
            ep_sq = board.ep_square
            if is_black:
                ep_sq ^= 56
            row, col = divmod(ep_sq, 8)
            planes[16, row, col] = 1.0

        # Player color (1.0 = white, 0.0 = black) — lets the network know
        # which side it is despite the symmetric encoding
        planes[17, :, :] = 0.0 if is_black else 1.0

        # Move count (normalized)
        planes[18, :, :] = min(board.fullmove_number / 200.0, 1.0)

        return torch.from_numpy(planes)

    def render(self, state: GameState) -> str:
        board = state.board
        lines = ["  a b c d e f g h"]
        for rank in range(7, -1, -1):
            row = [f"{rank + 1}"]
            for file in range(8):
                piece = board.piece_at(rank * 8 + file)
                row.append(piece.symbol() if piece else ".")
            row.append(f"{rank + 1}")
            lines.append(" ".join(row))
        lines.append("  a b c d e f g h")
        turn = "White" if board.turn == chess.WHITE else "Black"
        suffix = f"\n{turn} to move"
        if board.is_check():
            suffix += "  (check)"
        return "\n".join(lines) + suffix

    def action_to_san(self, state: GameState, action: int) -> str:
        """Convert an action index to Standard Algebraic Notation (e.g. 'Nf3')."""
        move = _action_to_move(action, state.board)
        if move is None:
            return f"<invalid action {action}>"
        return state.board.san(move)

    def parse_human_move(self, state: GameState, text: str) -> int | None:
        """Parse UCI ('e2e4') or SAN ('e4', 'Nf3', 'O-O') into a legal action index.

        Returns None if the input is unparseable or the move isn't legal.
        """
        text = text.strip()
        if not text:
            return None
        board = state.board
        move = None
        try:
            move = board.parse_uci(text)
        except ValueError:
            pass
        if move is None:
            try:
                move = board.parse_san(text)
            except ValueError:
                return None
        if move not in board.legal_moves:
            return None
        return _move_to_action(move, board.turn)

    def clone_state(self, state: GameState) -> GameState:
        return GameState(
            board=state.board.copy(),
            current_player=state.current_player,
            done=state.done,
            winner=state.winner,
        )
