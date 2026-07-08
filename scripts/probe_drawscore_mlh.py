"""Verify (a) draw_score is applied to the live leaf value, (b) the MLH gate
(moves_left_mcts) does NOT fire in the draw basin (Q below threshold), so it
isn't the length-collapse driver. CPU."""
import argparse, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory
from src.model.utils import wdl_to_scalar, support_to_scalar, inverse_scalar_transform
from scripts.eval_checkpoint_health import build_network


def load_buffer_games(path, game, limit=None):
    games = []
    with open(path, "rb") as f:
        header = pickle.load(f)
        version = header["version"]
        for i in range(header["n_records"]):
            record, priority = pickle.load(f)
            if version == 3:
                record = GameHistory.from_compact_dict(record, game)
            games.append(record)
            if limit and len(games) >= limit:
                break
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt")
    ap.add_argument("--buf", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    cfg = get_config("chess_small"); cfg.device = "cpu"
    HF = cfg.history_frames
    game = ChessGame()
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    net = build_network(ckpt, game, cfg, "cpu")
    print(f"net.draw_score={getattr(net,'draw_score',None)}  cfg.draw_score={cfg.draw_score}  "
          f"moves_left_mcts={getattr(cfg,'moves_left_mcts',None)}  ml_threshold={getattr(cfg,'ml_threshold',0.3)}")
    print(f"net has moves_left_head: {getattr(net,'moves_left_head',None) is not None}")

    games = load_buffer_games(args.buf, game, args.limit)
    rep = [g for g in games if g.draw_by_repetition]

    # (a) leaf value with/without draw_score on rep tail positions.
    print("\n[A] live leaf value on rep-tail positions (does draw_score apply?):")
    vals_ds, vals_no, mls = [], [], []
    with torch.no_grad():
        for g in rep[:30]:
            n = len(g)
            for ply in range(max(0, n - 6), n - 1):
                obs = g._stack_history(ply, HF)
                h = net.representation(obs.unsqueeze(0))
                _, vl = net.prediction(h)
                v_ds = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).item()
                v_no = wdl_to_scalar(vl.float(), draw_score=0.0).item()
                vals_ds.append(v_ds); vals_no.append(v_no)
                # also initial_inference path (what MCTS actually uses)
                if getattr(net, "moves_left_head", None) is not None:
                    ml_logits = net.predict_moves_left(h)
                    ml = inverse_scalar_transform(support_to_scalar(
                        ml_logits, net.moves_left_support_size)).clamp(min=0.0).item()
                    mls.append(ml)
    vds = np.array(vals_ds); vno = np.array(vals_no)
    print(f"   V(draw_score={cfg.draw_score}): mean={vds.mean():+.4f}  V(draw_score=0): mean={vno.mean():+.4f}")
    print(f"   mean shift from draw_score: {vds.mean()-vno.mean():+.4f}  (expect ~draw_score*mean_P_D)")

    # verify initial_inference returns the draw_score-adjusted scalar
    with torch.no_grad():
        obs = rep[0]._stack_history(len(rep[0]) - 2, HF)
        _, _, v_ii = net.initial_inference(obs.unsqueeze(0))
        h = net.representation(obs.unsqueeze(0)); _, vl = net.prediction(h)
        v_manual_ds = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).item()
        v_manual_no = wdl_to_scalar(vl.float(), draw_score=0.0).item()
    print(f"   initial_inference value={v_ii.item():+.4f}  manual(ds={cfg.draw_score})={v_manual_ds:+.4f}  "
          f"manual(ds=0)={v_manual_no:+.4f}  => MCTS leaf uses draw_score: {abs(v_ii.item()-v_manual_ds)<1e-4}")

    # (b) MLH gate: fraction of rep-tail Q (== leaf V magnitude) above ml_threshold.
    thr = float(getattr(cfg, "ml_threshold", 0.3))
    above = np.mean(np.abs(vno) > thr)
    print(f"\n[B] MLH gate (|Q|>{thr}) firing fraction at rep-tail: {above:.3f}")
    print(f"   => if ~0, the moves-left utility NEVER acts in the draw basin (not the length driver).")
    if mls:
        ml = np.array(mls)
        print(f"   moves_left head output at rep-tail: mean={ml.mean():.1f} plies (sanity: should be small near game end)")


if __name__ == "__main__":
    main()
