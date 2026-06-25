"""Syzygy tablebase probing at the MCTS root for MuZero self-play.

MuZero searches in latent space, so we can only probe the *real* board at the
root — but self-play visits every position as a root, so root-only probing
covers a full conversion (each ply of a KQ-K mate is its own root). This module
decodes the GPU batched state for the ≤max_pieces games and returns, per legal
move, the tablebase verdict (mover-POV value), DTZ-ranked among winning moves.
Consumed in tensor_mcts._select to overwrite the root children's value_score.

See tb_root_probe_plan_2026_06_25.md.
"""
import torch
import chess
import chess.syzygy

from src.games.chess import _action_to_move
from src.games.chess_gpu import CR_WK, CR_WQ, CR_BK, CR_BQ

ACTION_SPACE = 4672
_IDX_TO_PIECE = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def state_to_board(state, i: int) -> chess.Board:
    """Decode game ``i`` of a ChessBatchedState into a python-chess Board.

    Inverse of GpuChessGame.from_python_chess: planes 0-5 = white P,N,B,R,Q,K;
    6-11 = black; bit s set ⇒ square s occupied (python-chess square convention).
    """
    board = chess.Board.empty()
    pieces = state.pieces[i].tolist()
    for plane in range(12):
        bb = int(pieces[plane]) & 0xFFFFFFFFFFFFFFFF
        color = chess.WHITE if plane < 6 else chess.BLACK
        ptype = _IDX_TO_PIECE[plane % 6]
        for sq in chess.scan_forward(bb):
            board.set_piece_at(sq, chess.Piece(ptype, color))
    board.turn = chess.WHITE if int(state.side[i]) == 0 else chess.BLACK
    cr = state.castling[i].tolist()
    cf = ""
    if cr[CR_WK]:
        cf += "K"
    if cr[CR_WQ]:
        cf += "Q"
    if cr[CR_BK]:
        cf += "k"
    if cr[CR_BQ]:
        cf += "q"
    board.set_castling_fen(cf if cf else "-")
    ep = int(state.ep[i])
    board.ep_square = ep if ep >= 0 else None
    board.halfmove_clock = int(state.halfmove[i])
    return board


def _wdl_to_mover_value(child_wdl: int) -> float:
    """Map the CHILD's WDL (from the opponent's POV, since the move flips side)
    to OUR value. Syzygy WDL: +2 win / +1 cursed-win / 0 draw / -1 blessed-loss
    / -2 loss. We respect the 50-move rule, so cursed/blessed count as draws."""
    if child_wdl <= -2:   # opponent is lost ⇒ we win
        return 1.0
    if child_wdl >= 2:    # opponent wins ⇒ we lose
        return -1.0
    return 0.0            # draw / cursed-win / blessed-loss ⇒ draw


class SyzygyRootProber:
    """Probe the root board's legal moves against Syzygy; return an [N, A]
    mover-POV value tensor (NaN where not in TB range / not classifiable)."""

    def __init__(self, path: str, max_pieces: int = 5, dtz_weight: float = 0.05,
                 draw_score: float = 0.0):
        self.tb = chess.syzygy.open_tablebase(path)
        self.max_pieces = int(max_pieces)
        self.dtz_weight = float(dtz_weight)
        self.draw_score = float(draw_score)
        self._cache: dict[str, dict[int, float]] = {}
        self.last_in_tb = None   # [N] bool — games in TB range on the most recent call
        self.n_probed = 0        # cumulative count of root positions probed

    def close(self):
        try:
            self.tb.close()
        except Exception:
            pass

    @torch.no_grad()
    def root_move_values(self, state, legal_mask: torch.Tensor) -> torch.Tensor:
        """``legal_mask``: [N, A] bool. Returns [N, A] float32 (mover POV), NaN
        where the game isn't in TB range or a move can't be classified."""
        N = state.n
        dev = state.device
        out = torch.full((N, ACTION_SPACE), float("nan"), dtype=torch.float32)

        # GPU piece-count gate (cheap): popcount of the union of all piece
        # bitboards = number of occupied squares = piece count.
        occ = torch.zeros(N, dtype=torch.int64, device=dev)
        for p in range(12):
            occ = occ | state.pieces[:, p]
        bits = torch.zeros(N, dtype=torch.int64, device=dev)
        for b in range(64):
            bits = bits + ((occ >> b) & 1)
        in_tb = bits <= self.max_pieces
        self.last_in_tb = in_tb
        idxs = torch.nonzero(in_tb, as_tuple=False).flatten().tolist()
        self.n_probed += len(idxs)
        if not idxs:
            return out.to(dev)

        legal_cpu = legal_mask.cpu()
        for i in idxs:
            board = state_to_board(state, i)
            if not board.is_valid():
                continue
            key = board.fen()
            cached = self._cache.get(key)
            if cached is None:
                cached = self._classify(board, legal_cpu[i])
                self._cache[key] = cached
            for action, val in cached.items():
                out[i, action] = val
        return out.to(dev)

    def _classify(self, board: chess.Board, legal_row: torch.Tensor) -> dict[int, float]:
        """{action_index: mover_value} for the legal moves at ``board``,
        DTZ-ranked among winning moves (shortest → +1)."""
        vals: dict[int, float] = {}
        winners: list[tuple[int, int]] = []  # (action, |child_dtz|)
        for action in torch.nonzero(legal_row, as_tuple=False).flatten().tolist():
            move = _action_to_move(action, board)
            if move is None or move not in board.legal_moves:
                continue
            board.push(move)
            try:
                if board.is_checkmate():
                    vals[action] = 1.0
                elif board.is_stalemate() or board.is_insufficient_material():
                    vals[action] = self.draw_score
                else:
                    mv = _wdl_to_mover_value(self.tb.probe_wdl(board))
                    if mv > 0.0:
                        winners.append((action, abs(int(self.tb.probe_dtz(board)))))
                    else:
                        vals[action] = mv if mv < 0.0 else self.draw_score
            except (KeyError, ValueError, chess.syzygy.MissingTableError):
                pass  # leave NaN (net value used for this move)
            finally:
                board.pop()

        if winners:
            dz = [d for _, d in winners]
            dmin, dmax = min(dz), max(dz)
            span = max(1, dmax - dmin)
            for action, d in winners:
                vals[action] = 1.0 - self.dtz_weight * (d - dmin) / span
        return vals
