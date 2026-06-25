"""Root Syzygy tablebase probing in TensorMCTS.

With an untrained net the value head can't convert KQ-K. The TB override pins the
root children's value_score to the tablebase verdict, so the search piles visits
onto a *winning* conversion move instead of shuffling. Uses python-chess's bundled
tablebases copied to data/syzygy (KQvK etc.).
"""
import os
import numpy as np
import pytest
import torch
import chess

from src.config import get_config
from src.games.chess import ChessGame
from src.games.chess_gpu import GpuChessGame
from src.model.muzero_net import MuZeroNetwork
from src.mcts.tensor_mcts import TensorMCTS
from src.training.replay_buffer import stack_with_history

TB_DIR = "data/syzygy"
KQVK = "4k3/8/8/8/8/8/5Q2/4K3 w - - 0 1"   # white winning, white to move


def _net(A):
    net = MuZeroNetwork(
        observation_channels=22 * 8, action_space_size=A,
        hidden_planes=32, num_blocks=1, latent_h=8, latent_w=8,
        input_h=8, input_w=8, fc_hidden=32, value_support_size=10,
        value_head_type="wdl",
    )
    net.eval()
    return net


def _run(fen, tb_on):
    from src.games.syzygy_probe import SyzygyRootProber
    torch.manual_seed(0); np.random.seed(0)
    cg = ChessGame(); A = cg.action_space_size
    cstate = cg.reset_from_fen(fen)
    obs = stack_with_history(cg.to_tensor(cstate), [], 8).unsqueeze(0)

    gg = GpuChessGame()
    gstate = gg.from_python_chess([chess.Board(fen)], device="cpu")
    legal_mask = gg.legal_mask(gstate)
    if isinstance(legal_mask, tuple):
        legal_mask = legal_mask[0]

    root_tb = None
    if tb_on:
        prober = SyzygyRootProber(TB_DIR, max_pieces=5, dtz_weight=0.05)
        root_tb = prober.root_move_values(gstate, legal_mask)

    cfg = get_config("chess_small")
    cfg.num_simulations = 160
    cfg.device = "cpu"
    cfg.tb_root_probe = bool(tb_on)
    cfg.sample_k = A
    mcts = TensorMCTS(_net(A), cg, cfg, device="cpu", select_backend="eager")
    out = mcts.run_batch_gpu(obs, legal_mask, add_noise=False, root_tb_value=root_tb)
    acts = out["child_actions"][0].numpy()
    vis = out["child_visits"][0].numpy()
    return acts, vis, root_tb


@pytest.mark.skipif(not os.path.isdir(TB_DIR), reason="no data/syzygy tablebases")
def test_tb_root_steers_to_winning_move():
    acts, vis, root_tb = _run(KQVK, tb_on=True)
    row = root_tb[0]
    top = int(acts[int(np.argmax(vis))])
    # 1) the most-visited move is a tablebase WIN (value > 0), not a blunder.
    assert float(row[top]) > 0.0, f"top move tb-value={float(row[top])} should be winning"
    # 2) winning moves capture the large majority of visits.
    win_vis = sum(int(v) for a, v in zip(acts, vis) if float(row[int(a)]) > 0.0)
    assert win_vis / max(1, int(vis.sum())) > 0.8, "winning moves should dominate visits"


@pytest.mark.skipif(not os.path.isdir(TB_DIR), reason="no data/syzygy tablebases")
def test_tb_root_inert_when_off():
    # With the probe off the search is unchanged (no crash, plays normally).
    acts, vis, root_tb = _run(KQVK, tb_on=False)
    assert root_tb is None
    assert int(vis.sum()) > 0
