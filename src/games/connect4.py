"""Connect4 game implementation."""

import numpy as np
import torch

from .base import Game, GameState


class Connect4(Game):
    board_size = (6, 7)  # 6 rows, 7 columns
    action_space_size = 7  # drop piece in one of 7 columns
    num_planes = 3  # player1 pieces, player2 pieces, current player

    ROWS = 6
    COLS = 7

    def reset(self) -> GameState:
        return GameState(
            board=np.zeros((self.ROWS, self.COLS), dtype=np.int8),
            current_player=1,
        )

    def step(self, state: GameState, action: int) -> tuple[GameState, float, bool]:
        board = state.board.copy()
        col = action

        # Find lowest empty row in column
        row = -1
        for r in range(self.ROWS - 1, -1, -1):
            if board[r, col] == 0:
                row = r
                break
        assert row >= 0, f"Column {col} is full"

        board[row, col] = state.current_player

        # Check for win
        if self._check_win(board, row, col, state.current_player):
            return GameState(board=board, current_player=-state.current_player,
                             done=True, winner=state.current_player), 1.0, True

        # Check for draw
        if not np.any(board == 0):
            return GameState(board=board, current_player=-state.current_player,
                             done=True, winner=0), 0.0, True

        return GameState(board=board, current_player=-state.current_player), 0.0, False

    def legal_actions(self, state: GameState) -> list[int]:
        return [c for c in range(self.COLS) if state.board[0, c] == 0]

    def to_tensor(self, state: GameState) -> torch.Tensor:
        p1 = (state.board == 1).astype(np.float32)
        p2 = (state.board == -1).astype(np.float32)
        current = np.full((self.ROWS, self.COLS),
                          (state.current_player + 1) / 2, dtype=np.float32)
        return torch.from_numpy(np.stack([p1, p2, current]))

    def render(self, state: GameState) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = [" ".join(str(c) for c in range(self.COLS))]
        for row in state.board:
            lines.append(" ".join(symbols[c] for c in row))
        return "\n".join(lines)

    def _check_win(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """Check if the last move at (row, col) wins."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.ROWS and 0 <= c < self.COLS and board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False
