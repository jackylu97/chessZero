"""Checkers (8x8 American/English draughts) game implementation."""

import copy

import numpy as np
import torch

from .base import Game, GameState


# Pieces: positive = player 1 (bottom), negative = player 2 (top)
# 1 = man, 2 = king
EMPTY = 0
P1_MAN = 1
P1_KING = 2
P2_MAN = -1
P2_KING = -2


class Checkers(Game):
    board_size = (8, 8)
    num_planes = 6  # p1_men, p1_kings, p2_men, p2_kings, current_player, empty

    # Action encoding: from_square * 4 + direction_index for simple moves
    # + from_square * 4 + direction_index for jumps
    # 32 playable squares × 4 directions × 2 (move/jump) = 256 max actions
    # Simplified: we use from_sq * 32 + to_sq encoding (32 dark squares)
    # Actually, let's just use flat (from_row*8+from_col)*64 + (to_row*8+to_col)
    # But cap it to a reasonable size.
    # Simpler: enumerate all possible (from, to) on dark squares
    # 32 dark squares, max 4 targets each, ~128 simple + ~128 jump = 256
    # Let's use: action = from_dark_sq * 8 + move_index (up to 8 possible targets)
    action_space_size = 256  # 32 squares × 8 possible destinations

    # Map dark squares to indices 0-31
    _dark_squares = []
    _sq_to_idx = {}
    _idx_to_sq = {}

    def __init__(self):
        # Dark squares on standard checkers board (row+col is odd)
        idx = 0
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    Checkers._dark_squares.append((r, c))
                    Checkers._sq_to_idx[(r, c)] = idx
                    Checkers._idx_to_sq[idx] = (r, c)
                    idx += 1

    def reset(self) -> GameState:
        board = np.zeros((8, 8), dtype=np.int8)
        # Player 2 (top 3 rows)
        for r in range(3):
            for c in range(8):
                if (r + c) % 2 == 1:
                    board[r, c] = P2_MAN
        # Player 1 (bottom 3 rows)
        for r in range(5, 8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    board[r, c] = P1_MAN
        return GameState(board=board, current_player=1)

    def step(self, state: GameState, action: int) -> tuple[GameState, float, bool]:
        board = state.board.copy()
        from_idx = action // 8
        move_idx = action % 8
        from_sq = self._idx_to_sq[from_idx]

        # Find the actual move this action corresponds to
        moves = self._get_moves_for_piece(board, from_sq, state.current_player)
        if move_idx >= len(moves):
            raise ValueError(f"Invalid move index {move_idx} for piece at {from_sq}")

        to_sq, captured = moves[move_idx]
        piece = board[from_sq]
        board[from_sq] = EMPTY
        if captured:
            for cap in captured:
                board[cap] = EMPTY
        board[to_sq] = piece

        # King promotion
        if piece == P1_MAN and to_sq[0] == 0:
            board[to_sq] = P1_KING
        elif piece == P2_MAN and to_sq[0] == 7:
            board[to_sq] = P2_KING

        next_player = -state.current_player

        # Check if next player has any moves
        next_moves = self._all_legal_moves(board, next_player)
        if not next_moves:
            # Current player wins (opponent has no moves)
            return GameState(board=board, current_player=next_player,
                             done=True, winner=state.current_player), 1.0, True

        # Check if opponent has any pieces
        opp_pieces = (P2_MAN, P2_KING) if state.current_player == 1 else (P1_MAN, P1_KING)
        if not any(board[r, c] in opp_pieces for r in range(8) for c in range(8)):
            return GameState(board=board, current_player=next_player,
                             done=True, winner=state.current_player), 1.0, True

        return GameState(board=board, current_player=next_player), 0.0, False

    def legal_actions(self, state: GameState) -> list[int]:
        all_moves = self._all_legal_moves(state.board, state.current_player)
        actions = []
        for from_sq, to_sq, captured, move_idx in all_moves:
            from_idx = self._sq_to_idx[from_sq]
            actions.append(from_idx * 8 + move_idx)
        return actions

    def to_tensor(self, state: GameState) -> torch.Tensor:
        board = state.board
        planes = np.zeros((self.num_planes, 8, 8), dtype=np.float32)
        planes[0] = (board == P1_MAN).astype(np.float32)
        planes[1] = (board == P1_KING).astype(np.float32)
        planes[2] = (board == P2_MAN).astype(np.float32)
        planes[3] = (board == P2_KING).astype(np.float32)
        planes[4] = (state.current_player + 1) / 2  # 0 or 1
        planes[5] = (board == EMPTY).astype(np.float32)
        return torch.from_numpy(planes)

    def render(self, state: GameState) -> str:
        symbols = {
            EMPTY: ".", P1_MAN: "w", P1_KING: "W",
            P2_MAN: "b", P2_KING: "B",
        }
        lines = ["  0 1 2 3 4 5 6 7"]
        for r in range(8):
            row_str = " ".join(symbols[state.board[r, c]] for c in range(8))
            lines.append(f"{r} {row_str}")
        return "\n".join(lines)

    def _get_moves_for_piece(self, board: np.ndarray, sq: tuple[int, int],
                              player: int) -> list[tuple[tuple[int, int], list]]:
        """Get all moves for a piece. Returns list of (to_sq, captured_list)."""
        piece = board[sq]
        if piece == EMPTY:
            return []

        moves = []
        r, c = sq
        is_king = abs(piece) == 2

        # Directions: men move forward only, kings move both ways
        if player == 1:
            dirs = [(-1, -1), (-1, 1)]
            if is_king:
                dirs += [(1, -1), (1, 1)]
        else:
            dirs = [(1, -1), (1, 1)]
            if is_king:
                dirs += [(-1, -1), (-1, 1)]

        # Check jumps first (mandatory capture rule)
        jumps = self._get_jumps(board, sq, player, is_king)
        if jumps:
            return jumps

        # Simple moves
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == EMPTY:
                moves.append(((nr, nc), []))

        return moves

    def _get_jumps(self, board: np.ndarray, sq: tuple[int, int],
                    player: int, is_king: bool) -> list[tuple[tuple[int, int], list]]:
        """Get all jump sequences from a square."""
        r, c = sq
        if player == 1:
            dirs = [(-1, -1), (-1, 1)]
            if is_king:
                dirs += [(1, -1), (1, 1)]
            enemies = (P2_MAN, P2_KING)
        else:
            dirs = [(1, -1), (1, 1)]
            if is_king:
                dirs += [(-1, -1), (-1, 1)]
            enemies = (P1_MAN, P1_KING)

        jumps = []
        for dr, dc in dirs:
            mr, mc = r + dr, c + dc  # middle square
            lr, lc = r + 2 * dr, c + 2 * dc  # landing square
            if (0 <= lr < 8 and 0 <= lc < 8 and
                    board[mr, mc] in enemies and board[lr, lc] == EMPTY):
                jumps.append(((lr, lc), [(mr, mc)]))

        return jumps

    def _all_legal_moves(self, board: np.ndarray, player: int):
        """Get all legal moves for a player.

        Returns list of (from_sq, to_sq, captured, move_idx).
        If any jumps exist, only jumps are returned (mandatory capture).
        """
        if player == 1:
            pieces = (P1_MAN, P1_KING)
        else:
            pieces = (P2_MAN, P2_KING)

        all_jumps = []
        all_simple = []

        for r in range(8):
            for c in range(8):
                if board[r, c] in pieces:
                    sq = (r, c)
                    moves = self._get_moves_for_piece(board, sq, player)
                    for idx, (to_sq, captured) in enumerate(moves):
                        if captured:
                            all_jumps.append((sq, to_sq, captured, idx))
                        else:
                            all_simple.append((sq, to_sq, captured, idx))

        # Mandatory capture rule
        if all_jumps:
            return all_jumps
        return all_simple
