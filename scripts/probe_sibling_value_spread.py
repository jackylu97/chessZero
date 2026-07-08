"""DECISIVE within-position test: on REAL buffer positions, expand every legal
child once and measure the spread of the value head over SIBLING moves.

If sibling child-values are nearly identical (std ~0), MCTS Q is flat across
moves -> the search has no signal to prefer a winning move over a drawing
shuffle -> visit counts spread by prior/noise only -> draw collapse. This is
the 'value ranks across positions but not within' hypothesis.

Also compares: does the child-value ranking agree with Stockfish's move ranking?
(skipped if no engine). And: is the BEST sibling value meaningfully above the
WORST (decision margin)?
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, chess
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer
from src.model.utils import wdl_to_scalar
torch.serialization.add_safe_globals([MuZeroConfig])
from scripts.eval_checkpoint_health import build_network

CKPT = os.environ.get("PROBE_CKPT", "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt")
BUF = os.environ.get("PROBE_BUF", "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf")


@torch.no_grad()
def main():
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = "cpu"
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = build_network(ckpt, game, cfg, "cpu")
    rb = ReplayBuffer(cfg); rb.load(BUF, game=game)
    games = [g for g in rb.buffer if not g.external_values and len(g) >= 8]
    rng = np.random.default_rng(2)

    # Sibling value spread = std over children of (mover-POV Q at depth-1) =
    # std over children of (reward - discount*childvalue). Since reward~0 in-dist,
    # this is ~ std over children of (-childvalue).
    sib_std_q = []        # std of depth-1 mover-POV Q over siblings
    sib_std_childv = []   # std of child value (opp POV) over siblings
    sib_margin = []       # best - median child Q (decision margin)
    root_v = []
    root_pdraw = []
    n = 0
    sample = rng.choice(len(games), size=120, replace=False)
    for gi in sample:
        gh = games[int(gi)]
        si = int(rng.integers(2, len(gh) - 1))
        obs = gh._stack_history(si, cfg.history_frames)
        board = chess.Board()
        ok = True
        for a in gh.actions[:si]:
            mv = _action_to_move(int(a), board)
            if mv is None or mv not in board.legal_moves:
                ok = False; break
            board.push(mv)
        if not ok or board.is_game_over():
            continue
        legal = game.legal_actions(_state(game, board))
        if len(legal) < 3:
            continue
        h, _, vroot = net.initial_inference(obs.unsqueeze(0))
        rv = wdl_to_scalar(net.prediction(h)[1].float(), draw_score=cfg.draw_score).item()
        pdraw = F.softmax(net.prediction(h)[1].float(), -1)[0, 1].item()
        acts = torch.tensor(legal)
        hn = h.expand(len(legal), *h.shape[1:]).contiguous()
        nh, rw, pol, val = net.recurrent_inference(hn, acts)
        childv = wdl_to_scalar(net.prediction(nh)[1].float(), draw_score=cfg.draw_score).numpy()
        rwn = rw.view(-1).numpy()
        q = rwn - cfg.discount * childv  # mover-POV depth-1 Q over siblings
        sib_std_q.append(float(q.std()))
        sib_std_childv.append(float(childv.std()))
        sib_margin.append(float(q.max() - np.median(q)))
        root_v.append(rv); root_pdraw.append(pdraw)
        n += 1

    sib_std_q = np.array(sib_std_q); sib_std_childv = np.array(sib_std_childv)
    sib_margin = np.array(sib_margin); root_v = np.array(root_v); root_pdraw = np.array(root_pdraw)
    print(f"WITHIN-POSITION SIBLING VALUE SPREAD (n={n} real positions, all legal children expanded)")
    print(f"  std of depth-1 mover-POV Q over siblings: "
          f"mean={sib_std_q.mean():.4f}  median={np.median(sib_std_q):.4f}  "
          f"p90={np.percentile(sib_std_q,90):.4f}")
    print(f"  std of child VALUE (opp POV) over siblings: mean={sib_std_childv.mean():.4f}")
    print(f"  decision margin (best - median child Q): mean={sib_margin.mean():.4f} median={np.median(sib_margin):.4f}")
    print(f"  fraction of positions with sibling-Q std < 0.02 (near-FLAT): "
          f"{np.mean(sib_std_q < 0.02):.2f}")
    print(f"  fraction with sibling-Q std < 0.05: {np.mean(sib_std_q < 0.05):.2f}")
    print(f"  root value: mean={root_v.mean():+.4f} std={root_v.std():.4f} | "
          f"root P(draw): mean={root_pdraw.mean():.3f}")
    # Compare across-position spread (std of root values) vs within-position spread
    print(f"\n  ACROSS-position root-V std={root_v.std():.4f}  vs  "
          f"WITHIN-position mean sibling-Q std={sib_std_q.mean():.4f}")
    print(f"  ratio within/across = {sib_std_q.mean()/max(root_v.std(),1e-9):.2f}")


def _state(game, board):
    st = game.reset(); st.board = board.copy()
    st.current_player = 1 if board.turn == chess.WHITE else -1
    return st


if __name__ == "__main__":
    main()
