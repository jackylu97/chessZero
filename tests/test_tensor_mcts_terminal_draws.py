"""Root-terminal-draws override in TensorMCTS (the live GPU-resident engine).

When a root child's action completes a repetition, its selection value is pinned
to `draw_score` (mover POV). This is side-aware automatically: a winning side's
other children normalize above draw_score so it AVOIDS the repeating move; a
losing side's normalize below so it KEEPS the draw. We test the mechanism by
varying draw_score: a terrible draw (-1) suppresses the masked move, a great
draw (+1) makes the search pile onto it.
"""
import numpy as np
import torch

from src.config import get_config
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.mcts.tensor_mcts import TensorMCTS
from src.training.replay_buffer import stack_with_history


def _setup():
    g = ChessGame()
    A = g.action_space_size
    net = MuZeroNetwork(
        observation_channels=g.num_planes * 8, action_space_size=A,
        hidden_planes=32, num_blocks=1, latent_h=8, latent_w=8,
        input_h=8, input_w=8, fc_hidden=32, value_support_size=10,
        value_head_type="wdl",
    )
    net.eval()
    state = g.reset()
    legal = g.legal_actions(state)
    obs = stack_with_history(g.to_tensor(state), [], 8).unsqueeze(0)
    legal_mask = torch.zeros(1, A, dtype=torch.bool)
    legal_mask[0, legal] = True
    return g, A, net, obs, legal_mask, legal


def _run(g, A, net, obs, legal_mask, draw_score, fdm):
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = get_config("chess_small")
    cfg.num_simulations = 120
    cfg.device = "cpu"
    cfg.root_terminal_draws = True
    cfg.draw_score = draw_score
    cfg.sample_k = A
    mcts = TensorMCTS(net, g, cfg, device="cpu", select_backend="eager")
    out = mcts.run_batch_gpu(obs, legal_mask, add_noise=False, forced_draw_mask=fdm)
    acts = out["child_actions"][0].numpy()
    vis = out["child_visits"][0].numpy()
    return int(vis[acts == X_ACTION].sum())


X_ACTION = None  # set per-test


def test_terminal_draw_override_is_side_aware():
    global X_ACTION
    g, A, net, obs, legal_mask, legal = _setup()
    X_ACTION = legal[0]
    fdm = torch.zeros(1, A, dtype=torch.bool)
    fdm[0, X_ACTION] = True

    vx_none = _run(g, A, net, obs, legal_mask, -0.05, None)        # no override
    vx_bad = _run(g, A, net, obs, legal_mask, -1.0, fdm)           # draw is terrible
    vx_good = _run(g, A, net, obs, legal_mask, +1.0, fdm)          # draw is great

    # A terrible draw suppresses the repeating move; a great draw piles onto it.
    assert vx_bad < vx_none, f"bad-draw should suppress X: {vx_bad} !< {vx_none}"
    assert vx_good > vx_bad, f"good-draw should prefer X: {vx_good} !> {vx_bad}"
    assert vx_bad == 0, f"a -1 draw should fully avoid the repeat, got {vx_bad}"


def test_no_override_when_mask_none():
    """With forced_draw_mask=None the engine is unchanged (override inert)."""
    global X_ACTION
    g, A, net, obs, legal_mask, legal = _setup()
    X_ACTION = legal[0]
    a = _run(g, A, net, obs, legal_mask, -1.0, None)
    b = _run(g, A, net, obs, legal_mask, +1.0, None)
    # draw_score doesn't matter when no move is flagged as a repetition.
    assert a == b
