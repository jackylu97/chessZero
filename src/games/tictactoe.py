"""Tic-Tac-Toe game implementation."""

import functools
import numpy as np
import torch

from .base import Game, GameState


class TicTacToe(Game):
    board_size = (3, 3)
    action_space_size = 9
    num_planes = 3  # player1 pieces, player2 pieces, current player plane

    # All winning lines (rows, cols, diagonals)
    _WIN_LINES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6],  # diagonals
    ]

    def reset(self) -> GameState:
        return GameState(
            board=np.zeros(9, dtype=np.int8),
            current_player=1,
        )

    def step(self, state: GameState, action: int) -> tuple[GameState, float, bool]:
        board = state.board.copy()
        assert board[action] == 0, f"Illegal move: position {action} is occupied"
        board[action] = state.current_player

        # Check for win
        for line in self._WIN_LINES:
            if abs(board[line[0]] + board[line[1]] + board[line[2]]) == 3:
                new_state = GameState(
                    board=board,
                    current_player=-state.current_player,
                    done=True,
                    winner=state.current_player,
                )
                return new_state, 1.0, True

        # Check for draw
        if not np.any(board == 0):
            new_state = GameState(
                board=board,
                current_player=-state.current_player,
                done=True,
                winner=0,
            )
            return new_state, 0.0, True

        new_state = GameState(
            board=board,
            current_player=-state.current_player,
        )
        return new_state, 0.0, False

    def legal_actions(self, state: GameState) -> list[int]:
        return [i for i in range(9) if state.board[i] == 0]

    def to_tensor(self, state: GameState) -> torch.Tensor:
        board = state.board.reshape(3, 3)
        p1 = (board == 1).astype(np.float32)
        p2 = (board == -1).astype(np.float32)
        current = np.full((3, 3), (state.current_player + 1) / 2, dtype=np.float32)
        return torch.from_numpy(np.stack([p1, p2, current]))

    def render(self, state: GameState) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        board = state.board.reshape(3, 3)
        lines = ["  1 2 3"]
        for r, row in enumerate(board):
            cells = " ".join(symbols[c] for c in row)
            lines.append(f"{r + 1} {cells}")
        return "\n".join(lines)

    def render_with_moves(self, state: GameState) -> str:
        """Render board showing available move coordinates on empty squares."""
        symbols = {1: "X", -1: "O"}
        board = state.board.reshape(3, 3)
        lines = ["  1 2 3"]
        for r, row in enumerate(board):
            cells = []
            for c, val in enumerate(row):
                if val == 0:
                    cells.append(f"{r + 1}{c + 1}")  # e.g. "11", "23"
                else:
                    cells.append(f" {symbols[val]} ")
            lines.append(f"{r + 1} {'|'.join(cells)}")
        return "\n".join(lines)


class MinimaxPlayer:
    """Perfect tic-tac-toe player via minimax with alpha-beta pruning.

    The full game tree is tiny (~5477 terminal nodes) so results are cached
    after the first call — subsequent lookups are O(1).
    """

    _WIN_LINES = TicTacToe._WIN_LINES

    def best_action(self, board: np.ndarray, player: int) -> int:
        """Return the optimal action for `player` given flat `board`."""
        _, action = self._minimax(tuple(board), player, -2, 2)
        return action

    @functools.lru_cache(maxsize=None)
    def _minimax(self, board: tuple, player: int, alpha: float, beta: float) -> tuple[float, int]:
        score = self._terminal_score(board)
        if score is not None:
            return score, -1

        best_action = -1
        best_score = -2 * player  # -inf for maximiser, +inf for minimiser

        for action in range(9):
            if board[action] != 0:
                continue
            next_board = list(board)
            next_board[action] = player
            score, _ = self._minimax(tuple(next_board), -player, alpha, beta)

            if player == 1:
                if score > best_score:
                    best_score, best_action = score, action
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score, best_action = score, action
                beta = min(beta, best_score)

            if beta <= alpha:
                break

        return best_score, best_action

    def _terminal_score(self, board: tuple) -> float | None:
        for line in self._WIN_LINES:
            s = board[line[0]] + board[line[1]] + board[line[2]]
            if s == 3:
                return 1.0
            if s == -3:
                return -1.0
        if all(b != 0 for b in board):
            return 0.0
        return None
