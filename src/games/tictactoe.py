"""Tic-Tac-Toe game implementation."""

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
        lines = []
        for row in board:
            lines.append(" ".join(symbols[c] for c in row))
        return "\n".join(lines)
