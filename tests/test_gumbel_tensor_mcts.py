"""Tests for the Plain Gumbel MuZero root in TensorMCTS (port of the numpy
BatchedMCTS implementation; mctx-style considered-visit-level formulation).

Covers:
- Sequential Halving level table structure.
- End-to-end gumbel search on a real (random-weight) chess net: legality,
  target normalization, visit budget accounting, candidate containment.
- Noise-off determinism.
- ORACLE PARITY: noise-off tensor gumbel vs the numpy BatchedMCTS gumbel on
  identical inputs — same chosen action, π' targets allclose. The two engines'
  PUCT/backup cores were previously verified identical to ~1e-16, so the
  gumbel layers should agree closely too.
"""
import numpy as np
import pytest
import torch

from src.config import get_config
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _tiny_net(game, cfg):
    torch.manual_seed(7)
    return MuZeroNetwork(
        observation_channels=game.num_planes,  # tests feed single frames
        action_space_size=game.action_space_size,
        hidden_planes=32, num_blocks=1, latent_h=8, latent_w=8,
        input_h=8, input_w=8, fc_hidden=32,
        value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size,
        action_embed_dim=cfg.action_embed_dim,
        value_head_type="wdl", policy_head_type=cfg.policy_head_type,
    ).to(DEV).eval()


def _unzero_policy(net):
    """The policy head is zero-initialized (uniform policy), which makes ALL
    root logits tie — candidate top-m becomes an arbitrary tie-break and the
    numpy/tensor engines legitimately pick different sets. Give the head
    random weights so logits are distinct and parity is well-defined."""
    torch.manual_seed(11)
    head = net.prediction.policy_head
    proj = getattr(head, "proj", None)
    if proj is not None:
        torch.nn.init.normal_(proj.weight, std=0.05)
    return net


def _gumbel_cfg(num_sims=32, m=16):
    cfg = get_config("chess_small")
    cfg.use_gumbel = True
    cfg.use_gumbel_noise = False       # deterministic: g == 0
    cfg.gumbel_num_considered = m
    cfg.num_simulations = num_sims
    cfg.moves_left_mcts = False
    cfg.tb_root_probe = False
    cfg.root_terminal_draws = False
    return cfg


def _obs_and_legal(game, fens=None):
    import chess
    states = ([game.reset(), game.reset_from_fen("3k4/8/3K4/8/8/8/8/3Q4 w - - 0 1")]
              if fens is None else [game.reset_from_fen(f) for f in fens])
    obs = torch.stack([game.to_tensor(s) for s in states]).to(DEV)
    legal_lists = [game.legal_actions(s) for s in states]
    legal_mask = torch.zeros(len(states), game.action_space_size,
                             dtype=torch.bool, device=DEV)
    for i, ll in enumerate(legal_lists):
        legal_mask[i, ll] = True
    return obs, legal_mask, legal_lists


def test_level_table_structure():
    from src.mcts.tensor_mcts import TensorMCTS
    tbl = TensorMCTS._gumbel_level_table(32, 16)
    assert tbl == [0] * 16 + [1] * 8 + [2] * 4 + [3] * 4
    tbl = TensorMCTS._gumbel_level_table(400, 16)
    assert len(tbl) == 400 and tbl[:16] == [0] * 16


@pytest.mark.skipif(DEV == "cpu", reason="tensor MCTS targets CUDA")
def test_gumbel_search_structure_and_determinism():
    from src.mcts.tensor_mcts import TensorMCTS
    game = ChessGame()
    cfg = _gumbel_cfg()
    net = _tiny_net(game, cfg)
    obs, legal_mask, legal_lists = _obs_and_legal(game)

    def run():
        mcts = TensorMCTS(net, game, cfg, device=DEV,
                          hidden_dtype=torch.float32, select_backend="eager")
        return mcts.run_batch_gpu(obs.clone(), legal_mask.clone(), add_noise=True)

    rd = run()
    A = game.action_space_size
    act = rd["gumbel_action"].cpu().numpy()
    pol = rd["gumbel_policy"].cpu().numpy()
    for i, ll in enumerate(legal_lists):
        assert int(act[i]) in ll, "chosen action must be legal"
        assert abs(pol[i].sum() - 1.0) < 1e-4
        illegal = np.setdiff1d(np.arange(A), np.asarray(ll))
        assert pol[i][illegal].max() == 0.0, "π' must be zero on illegal actions"
    # Visit budget: all root visits live on candidate slots and sum to num_sims.
    visits = rd["child_visits"].cpu().numpy()
    actions = rd["child_actions"].cpu().numpy()
    for i in range(2):
        assert visits[i][actions[i] == -1].sum() == 0
        assert visits[i].sum() == cfg.num_simulations
    # Determinism (noise off): identical outputs across runs.
    rd2 = run()
    assert np.array_equal(act, rd2["gumbel_action"].cpu().numpy())
    assert np.allclose(pol, rd2["gumbel_policy"].cpu().numpy(), atol=1e-6)


@pytest.mark.skipif(DEV == "cpu", reason="tensor MCTS targets CUDA")
def test_gumbel_with_noise_produces_valid_actions():
    """Regression: the Gumbel(0,1) draw had an operator-precedence bug
    (clamp before unary minus) that NaN'd every perturbed logit — all
    candidates invalid, action = -1 at every ply. Noise-ON must yield legal
    actions and finite normalized targets."""
    from src.mcts.tensor_mcts import TensorMCTS
    game = ChessGame()
    cfg = _gumbel_cfg()
    cfg.use_gumbel_noise = True
    net = _tiny_net(game, cfg)
    obs, legal_mask, legal_lists = _obs_and_legal(game)
    torch.manual_seed(0)
    mcts = TensorMCTS(net, game, cfg, device=DEV,
                      hidden_dtype=torch.float32, select_backend="eager")
    rd = mcts.run_batch_gpu(obs, legal_mask, add_noise=True)
    act = rd["gumbel_action"].cpu().numpy()
    pol = rd["gumbel_policy"].cpu().numpy()
    for i, ll in enumerate(legal_lists):
        assert int(act[i]) in ll, f"noise-on action {act[i]} not legal"
        assert np.isfinite(pol[i]).all()
        assert abs(pol[i].sum() - 1.0) < 1e-4
    # Perturbed scores must be finite on candidate slots.
    st = mcts._gumbel_state
    cand = mcts.child_actions[:, 0, :st["m"]].cpu().numpy()
    pert = st["perturbed"].cpu().numpy()
    assert np.isfinite(pert[cand != -1]).all(), "NaN/inf in perturbed logits"


@pytest.mark.skipif(DEV == "cpu", reason="tensor MCTS targets CUDA")
def test_gumbel_matches_numpy_oracle():
    from src.mcts.mcts import BatchedMCTS, select_action_gumbel
    from src.mcts.tensor_mcts import TensorMCTS
    game = ChessGame()
    cfg = _gumbel_cfg(num_sims=32, m=8)
    net = _unzero_policy(_tiny_net(game, cfg))
    obs, legal_mask, legal_lists = _obs_and_legal(game)

    t_mcts = TensorMCTS(net, game, cfg, device=DEV,
                        hidden_dtype=torch.float32, select_backend="eager")
    rd = t_mcts.run_batch_gpu(obs, legal_mask, add_noise=True)
    t_act = rd["gumbel_action"].cpu().numpy()
    t_pol = rd["gumbel_policy"].cpu().numpy()

    n_mcts = BatchedMCTS(net, game, cfg, DEV)
    roots = n_mcts.run_batch([obs[i].cpu() for i in range(2)],
                             legal_lists, add_noise=True)
    for i, root in enumerate(roots):
        a_np, pi_np = select_action_gumbel(root, cfg, game.action_space_size)
        assert int(t_act[i]) == int(a_np), (
            f"game {i}: tensor chose {t_act[i]}, numpy oracle {a_np}")
        np.testing.assert_allclose(t_pol[i], pi_np, atol=2e-3,
                                   err_msg=f"game {i} π' mismatch")
