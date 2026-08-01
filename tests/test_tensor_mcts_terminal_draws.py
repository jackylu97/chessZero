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


# --------------------------------------------------------------------------
# Gumbel path: the veto must reach the π' TARGET and A_{n+1}, not just visits.
#
# 2026-07-22 regression suite: _gumbel_finalize used to ignore
# _root_term_mask and _expand backed up the network's value for pinned
# children — the veto was a visit-allocation nudge while the stored policy
# target kept full phantom-win mass on the drawing move (buffer audit:
# 14-25% mass on vetoed moves outside TB, threefold still 6% of organic
# games with the veto "on"). These tests use a deterministic phantom-win
# stub (every root Q = +0.9, the repetition-shuffle regime) and assert on
# gumbel_policy / gumbel_action / the terminal 0.0 backup — the quantities
# the visit-only tests above cannot see.
# --------------------------------------------------------------------------

_FIXED_LOGITS = None


class _PhantomStub:
    """Parity-in-hidden stub: value = 0.9 * root_sign from the root STM's POV
    at every node (sign alternates with depth via hidden[:, 0, 0, 0]).
    root_sign=+1 → root is winning everywhere (phantom-win regime);
    root_sign=-1 → root is losing everywhere. Rewards 0, fixed logits."""

    def __init__(self, A: int, root_sign: float = 1.0):
        global _FIXED_LOGITS
        if _FIXED_LOGITS is None or _FIXED_LOGITS.shape[0] != A:
            gen = torch.Generator().manual_seed(3)
            _FIXED_LOGITS = torch.randn(A, generator=gen) * 0.5
        self.A = A
        self.root_sign = float(root_sign)

    def parameters(self):
        return iter(())

    def initial_inference(self, obs):
        n = obs.shape[0]
        hidden = torch.zeros(n, 2, 8, 8)
        hidden[:, 0, 0, 0] = 1.0
        pol = _FIXED_LOGITS.unsqueeze(0).expand(n, self.A).clone()
        val = torch.full((n,), 0.9 * self.root_sign)
        return hidden, pol, val

    def recurrent_inference(self, hidden, actions):
        n = hidden.shape[0]
        parity = hidden[:, 0, 0, 0]
        nxt = hidden.clone()
        nxt[:, 0, 0, 0] = -parity
        pol = _FIXED_LOGITS.unsqueeze(0).expand(n, self.A).clone()
        val = 0.9 * self.root_sign * (-parity)
        rew = torch.zeros(n)
        return nxt, rew, pol, val


def _gumbel_cfg(draw_score=-0.05, veto=True):
    cfg = get_config("chess_small")
    cfg.use_gumbel = True
    cfg.use_gumbel_noise = False
    cfg.gumbel_num_considered = 8
    cfg.num_simulations = 60
    cfg.moves_left_mcts = False
    cfg.tb_root_probe = False
    cfg.root_terminal_draws = veto
    cfg.draw_score = draw_score
    cfg.sample_k = 50
    cfg.device = "cpu"
    return cfg


def _gumbel_setup(root_sign=1.0):
    g = ChessGame()
    A = g.action_space_size
    net = _PhantomStub(A, root_sign=root_sign)
    state = g.reset_from_fen("3k4/8/3K4/8/8/8/8/3Q4 w - - 0 1")
    obs = g.to_tensor(state).unsqueeze(0)
    legal = g.legal_actions(state)
    legal_mask = torch.zeros(1, A, dtype=torch.bool)
    legal_mask[0, legal] = True
    return g, A, net, obs, legal_mask, legal


def _run_tensor_gumbel(g, net, cfg, obs, legal_mask, fdm):
    torch.manual_seed(0)
    np.random.seed(0)
    m = TensorMCTS(net, g, cfg, device="cpu", select_backend="eager")
    out = m.run_batch_gpu(obs.clone(), legal_mask.clone(), add_noise=False,
                          forced_draw_mask=fdm)
    return m, out


def test_gumbel_veto_pi_prime_action_and_no_expand():
    """Winning side + veto: π'(F)≈0, A_{n+1}≠F, and F is a TRUE terminal —
    never expanded, only 0.0 ever backed up into its stats."""
    g, A, net, obs, legal_mask, _ = _gumbel_setup(root_sign=1.0)

    _, base = _run_tensor_gumbel(g, net, _gumbel_cfg(veto=False), obs, legal_mask, None)
    F = int(base["gumbel_action"][0])
    assert base["gumbel_policy"][0, F] > 0.05  # F is a live candidate without the veto

    fdm = torch.zeros(1, A, dtype=torch.bool)
    fdm[0, F] = True
    m, out = _run_tensor_gumbel(g, net, _gumbel_cfg(veto=True), obs, legal_mask, fdm)

    assert int(out["gumbel_action"][0]) != F, "A_{n+1} still plays the vetoed draw"
    assert float(out["gumbel_policy"][0, F]) < 1e-3, (
        f"π' keeps phantom-win mass on the vetoed move: {float(out['gumbel_policy'][0, F]):.4f}"
    )
    # Terminal semantics: pinned child never expanded; value_sum is pure 0.0
    # draw backups (any nonzero = phantom network value leaked in).
    root_acts = m.child_actions[0, 0, :]
    slot = int((root_acts == F).nonzero()[0])
    assert int(m.child_node_idx[0, 0, slot]) == -1, "vetoed child was expanded"
    assert float(m.child_value_sum[0, 0, slot]) == 0.0, "phantom value backed up"
    assert int(m.child_visits[0, 0, slot]) > 0  # SH level-0 did visit it


def test_gumbel_veto_numpy_parity():
    """Tensor engine must match the numpy oracle under the veto: same action,
    same π' (within fp tolerance), same root value."""
    from src.mcts.mcts import BatchedMCTS, select_action_gumbel

    g, A, net, obs, legal_mask, legal = _gumbel_setup(root_sign=1.0)
    _, base = _run_tensor_gumbel(g, net, _gumbel_cfg(veto=False), obs, legal_mask, None)
    F = int(base["gumbel_action"][0])

    fdm = torch.zeros(1, A, dtype=torch.bool)
    fdm[0, F] = True
    cfg = _gumbel_cfg(veto=True)
    _, out = _run_tensor_gumbel(g, net, cfg, obs, legal_mask, fdm)

    torch.manual_seed(0)
    np.random.seed(0)
    bm = BatchedMCTS(net, g, cfg, "cpu")
    root = bm.run_batch([obs[0]], [legal], add_noise=False,
                        forced_draw_actions=[{F}])[0]
    n_act, n_pi = select_action_gumbel(root, cfg, A)

    assert int(out["gumbel_action"][0]) == int(n_act)
    np.testing.assert_allclose(
        out["gumbel_policy"][0].numpy(), n_pi, atol=2e-3,
        err_msg="tensor π' diverges from numpy under the veto")
    assert abs(float(out["root_value"][0]) - float(root.value)) < 1e-3


def test_gumbel_veto_losing_side_keeps_draw():
    """Losing side (all alternatives Q≈-0.9): the 0.0 draw is the BEST option —
    the veto must not mask it. π' concentrates on F and A_{n+1} takes it."""
    g, A, net, obs, legal_mask, legal = _gumbel_setup(root_sign=-1.0)

    # Veto the no-veto favorite so F is guaranteed to be a Gumbel candidate
    # (the veto only pins sampled root children — numpy parity).
    _, base = _run_tensor_gumbel(g, net, _gumbel_cfg(veto=False), obs, legal_mask, None)
    F = int(base["gumbel_action"][0])
    fdm = torch.zeros(1, A, dtype=torch.bool)
    fdm[0, F] = True
    _, out = _run_tensor_gumbel(g, net, _gumbel_cfg(veto=True), obs, legal_mask, fdm)

    assert int(out["gumbel_action"][0]) == F, "losing side must keep its draw resource"
    assert float(out["gumbel_policy"][0, F]) > 0.5, (
        "draw should dominate π' when every alternative loses")


def test_puct_veto_terminal_backup_is_zero():
    """PUCT path: even when selection PILES onto the pinned child
    (draw_score=+1), its value_sum stays exactly 0.0 — every visit backed up
    the draw, never the network's value — and it is never expanded."""
    g, A, net, obs, legal_mask, legal = _gumbel_setup(root_sign=1.0)
    F = legal[0]
    fdm = torch.zeros(1, A, dtype=torch.bool)
    fdm[0, F] = True

    cfg = _gumbel_cfg(draw_score=+1.0, veto=True)
    cfg.use_gumbel = False
    torch.manual_seed(0)
    np.random.seed(0)
    m = TensorMCTS(net, g, cfg, device="cpu", select_backend="eager")
    m.run_batch_gpu(obs.clone(), legal_mask.clone(), add_noise=False,
                    forced_draw_mask=fdm)
    root_acts = m.child_actions[0, 0, :]
    slot = int((root_acts == F).nonzero()[0])
    assert int(m.child_visits[0, 0, slot]) > 10, "draw_score=+1 should attract visits"
    assert float(m.child_value_sum[0, 0, slot]) == 0.0, "phantom value backed up"
    assert float(m.child_rewards[0, 0, slot]) == 0.0, "phantom reward mirrored"
    assert int(m.child_node_idx[0, 0, slot]) == -1, "vetoed child was expanded"
