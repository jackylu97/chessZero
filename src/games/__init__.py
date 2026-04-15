from .base import Game, GameState
from .tictactoe import TicTacToe
from .connect4 import Connect4
from .chess import ChessGame
from .checkers import Checkers

GAME_REGISTRY = {
    "tictactoe": TicTacToe,
    "connect4": Connect4,
    "chess": ChessGame,
    "checkers": Checkers,
}
