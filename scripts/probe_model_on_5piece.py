"""What does the model OUTPUT at 5-piece positions? The buffer only has DTZ=1
five-piece positions (the probe steers the immediate simplifying capture), so we
can't measure 5-piece DTZ-ranking from it. Generate DTZ-diverse winning 5-piece
positions and read the model's value head + no-probe policy argmax directly.

  value corr(val,-DTZ): does the value rank 5-piece positions by distance to mate?
  value win-recognition: does it even call them winning (>0)?
  policy win-preserve / DTZ-optimal: unaided, does it pick a winning / best move?

NOTE: generated positions get zero history (the net uses 8 frames) — absolute
values may be off-distribution, but the ranking/move-choice signal is informative.

Run: PYTHONPATH=. .venv/bin/python scripts/probe_model_on_5piece.py --checkpoint <ckpt.pt>
"""
import argparse, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, chess, chess.syzygy
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.training.replay_buffer import stack_with_history
from scripts.eval_checkpoint_health import build_network

PIECE = {'Q': chess.QUEEN, 'R': chess.ROOK, 'B': chess.BISHOP, 'N': chess.KNIGHT, 'P': chess.PAWN}
# winning-side(white) extra vs black extra; total extras = 3 -> 5 pieces with the 2 kings
# pawnless winning 5-piece material -> maneuvering required -> DTZ>1 (forces a real
# distance-to-mate gradient, unlike pawn endgames which are ~always DTZ=1).
CONFIGS = [(['Q', 'Q'], ['R']), (['R', 'R'], ['N']), (['R', 'R'], ['B']), (['Q'], ['N', 'N']),
           (['Q', 'R'], ['Q']), (['B', 'B'], ['N']), (['Q'], ['B', 'B']), (['R', 'B'], ['N']),
           (['Q', 'B'], ['R']), (['Q', 'N'], ['R'])]
DZB = [(1, 1), (2, 5), (6, 12), (13, 30), (31, 90)]


def gen_positions(tb, n_per_bucket, seed=0):
    rng = random.Random(seed); cells = {b: [] for b in DZB}; tries = 0
    while tries < 800000 and not all(len(cells[b]) >= n_per_bucket for b in DZB):
        tries += 1
        we, be = rng.choice(CONFIGS)
        sqs = rng.sample(range(64), 2 + len(we) + len(be))
        if chess.square_distance(sqs[0], sqs[1]) <= 1:
            continue
        board = chess.Board.empty()
        board.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
        ok = True; i = 2
        for sym in we + be:
            sq = sqs[i]; i += 1
            if sym == 'P' and chess.square_rank(sq) in (0, 7):
                ok = False; break
            color = chess.WHITE if sym in [s for s in we] and i - 1 < 2 + len(we) else chess.BLACK
            board.set_piece_at(sq, chess.Piece(PIECE[sym], color))
        if not ok:
            continue
        board.turn = chess.WHITE
        if not board.is_valid() or board.is_game_over():
            continue
        try:
            if tb.probe_wdl(board) != 2:
                continue
            d = abs(int(tb.probe_dtz(board)))
        except Exception:
            continue
        for (lo, hi) in DZB:
            if lo <= d <= hi and len(cells[(lo, hi)]) < n_per_bucket:
                cells[(lo, hi)].append((board.fen(), d)); break
    return [p for c in cells.values() for p in c]


def best_dtz_set(board, tb):
    """win-preserving moves and the DTZ-optimal subset."""
    keep, dz = {}, {}
    for mv in board.legal_moves:
        board.push(mv)
        try:
            k = board.is_checkmate() or (tb.probe_wdl(board) < 0)
            d = 0 if board.is_checkmate() else abs(int(tb.probe_dtz(board)))
        except Exception:
            k = False; d = None
        board.pop()
        if k:
            keep[mv] = d
    best = min([d for d in keep.values() if d is not None], default=None)
    opt = {mv for mv, d in keep.items() if d == best} if best is not None else set()
    return set(keep), opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tb", default="data/syzygy")
    ap.add_argument("--per-bucket", type=int, default=60)
    args = ap.parse_args()
    dev = "cpu"
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = dev; hf = cfg.history_frames
    torch.serialization.add_safe_globals([MuZeroConfig])
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ck, game, cfg, dev); net.eval()
    tb = chess.syzygy.open_tablebase(args.tb)
    pos = gen_positions(tb, args.per_bucket)
    print(f"generated {len(pos)} DTZ-diverse winning 5-piece positions "
          f"(buckets {[sum(1 for _,d in pos if lo<=d<=hi) for lo,hi in DZB]})\n")

    vals, dtzs, pres, opt, vpos = [], [], 0, 0, 0
    with torch.no_grad():
        for fen, d in pos:
            board = chess.Board(fen)
            cur = game.to_tensor(game.reset_from_fen(fen))
            obs = stack_with_history(cur, [], hf).unsqueeze(0).to(dev)
            _, logits, value = net.initial_inference(obs)
            v = float(value.item()); vals.append(v); dtzs.append(float(d))
            if v > 0:
                vpos += 1
            lg = logits[0].cpu().numpy()
            la = {}
            for mv in board.legal_moves:
                la[_move_to_action(mv, board.turn)] = mv
            mask = np.full(lg.shape, -1e9)
            for ai in la:
                if 0 <= ai < lg.shape[0]:
                    mask[ai] = lg[ai]
            am = int(np.argmax(mask)); mv = la.get(am)
            if mv is None:
                continue
            keep, optset = best_dtz_set(board, tb)
            if mv in keep:
                pres += 1
            if mv in optset:
                opt += 1
    tb.close()
    vals = np.array(vals); dtzs = np.array(dtzs); n = len(vals)
    corr = np.corrcoef(vals, -dtzs)[0, 1] if dtzs.std() > 1e-6 else float("nan")
    print(f"=== model on {n} generated 5-piece WINNING positions (step {ck.get('step','?')}) ===")
    print(f"  VALUE: mean {vals.mean():+.3f}  | calls them winning (val>0): {vpos/n:.1%}")
    print(f"  VALUE ranks by DTZ: corr(val,-DTZ) = {corr:+.3f}   (want strongly +)")
    print(f"  POLICY (no probe) argmax move PRESERVES win: {pres/n:.1%}")
    print(f"  POLICY (no probe) argmax move is DTZ-OPTIMAL: {opt/n:.1%}")


if __name__ == "__main__":
    main()
