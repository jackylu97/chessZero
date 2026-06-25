"""Does the root TB probe actually CONVERT, or just avoid losing the win?
Play an elementary endgame with the probe-steered model and report plies-to-mate.
If it mates fast -> the probe converts. If it shuffles to the cap -> the DTZ
progress gradient is too weak (avoids losing the win but doesn't make progress).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, chess, chess.syzygy

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.games.chess_gpu import GpuChessGame
from src.mcts.tensor_mcts import TensorMCTS
from src.games.syzygy_probe import SyzygyRootProber
from src.training.replay_buffer import stack_with_history
from scripts.eval_checkpoint_health import build_network

POSITIONS = {
    "KQvK": "4k3/8/8/8/8/8/5Q2/4K3 w - - 0 1",
    "KRvK": "4k3/8/8/8/8/8/5R2/4K3 w - - 0 1",
}


def play(net, cg, gpu, game_dev, fen, dtz_weight, sims, max_plies=120, hard=False):
    cfg = get_config("chess_small"); cfg.device = game_dev
    cfg.num_simulations = sims; cfg.tb_root_probe = True
    cfg.tensor_mcts_select_backend = "eager"
    mcts = TensorMCTS(net, cg, cfg, device=game_dev, select_backend="eager")
    prober = SyzygyRootProber("data/syzygy", max_pieces=5, dtz_weight=dtz_weight)
    tb = chess.syzygy.open_tablebase("data/syzygy")
    board = chess.Board(fen)
    frames = []
    dtz0 = abs(tb.probe_dtz(board))
    for ply in range(max_plies):
        if board.is_game_over():
            break
        cstate = cg.reset_from_fen(board.fen())
        cur = cg.to_tensor(cstate)
        obs = stack_with_history(cur, frames, 8).unsqueeze(0).to(game_dev)
        gstate = gpu.from_python_chess([board], device=game_dev)
        legal_mask = gpu.legal_mask(gstate)
        if isinstance(legal_mask, tuple):
            legal_mask = legal_mask[0]
        root_tb = prober.root_move_values(gstate, legal_mask)
        out = mcts.run_batch_gpu(obs, legal_mask, add_noise=False, root_tb_value=root_tb)
        # hard=True: hard-select the DTZ-optimal TB move (one-hot, KLD-risky).
        # hard=False: soft — take the search's argmax-visit move (the value bias
        # shapes the visits; policy target stays the visit distribution → KLD-safe).
        ba = int(prober.last_best_action[0]) if prober.last_best_action is not None else -1
        if hard and ba >= 0:
            action = ba
        else:
            acts = out["child_actions"][0]; vis = out["child_visits"][0]
            action = int(acts[int(vis.argmax())])
        move = _action_to_move(action, board)
        if move is None or move not in board.legal_moves:
            return f"illegal move at ply {ply}", ply, dtz0, None
        frames.append(cur)
        board.push(move)
    tb.close(); prober.close()
    res = "MATE" if board.is_checkmate() else ("draw50" if board.halfmove_clock >= 100
            else "stalemate" if board.is_stalemate() else f"unfinished({board.result()})")
    return res, ply, dtz0, board.fen()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--dtz-weights", type=float, nargs="+", default=[0.05, 0.5])
    ap.add_argument("--hard", action="store_true", help="hard-select TB move (default: soft).")
    ap.add_argument("--max-plies", type=int, default=120)
    args = ap.parse_args()
    dev = args.device
    cg = ChessGame(); gpu = GpuChessGame()
    cfg = get_config("chess_small")
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, cg, cfg, dev); net.eval()
    print(f"net step {ckpt.get('step','?')} | sims {args.sims}\n")
    print(f"{'position':>8} {'dtz_w':>6} {'result':>16} {'plies':>6} {'startDTZ':>9}")
    for name, fen in POSITIONS.items():
        for w in args.dtz_weights:
            res, ply, dtz0, final = play(net, cg, gpu, dev, fen, w, args.sims,
                                          max_plies=args.max_plies, hard=args.hard)
            print(f"{name:>8} {w:>6.2f} {res:>16} {ply:>6} {dtz0:>9}")
    print("\n  MATE in ~startDTZ plies => probe converts. unfinished/draw50 => it shuffles "
          "(DTZ gradient too weak).")


if __name__ == "__main__":
    main()
