"""Regression test for the Sampled-MuZero root/leaf prior bug in TensorMCTS.

Bug (fixed): ``src/mcts/tensor_mcts.py`` populated K=sample_k DUPLICATE sampled
slots, each at a uniform 1/K prior (Option (b)), and PUCT did a PER-SLOT argmax.
A popular action's prior mass was fragmented across its duplicate slots (each
~0.02) instead of summed to count(a)/K on a single entry. Once any child is
visited the value_score (∈[0,1]) swamps the ~0.02 prior and the search collapses
onto one move — exploring a mean of ~2.4 distinct actions and putting 88-100% of
visits on a single move, vs ~9.7 for the correct numpy paths.

The fix dedups the K i.i.d. samples to UNIQUE actions with prior β̂(a)=count(a)/K
on one slot each (padding slots = -1 / prior 0), matching the numpy oracle
(``mcts.py`` BatchedMCTS + ``_sample_iid_with_replacement``) at BOTH root and leaf.

This test runs TensorMCTS and BatchedMCTS (the numpy oracle) with the SAME small
real network on several real chess positions and asserts TensorMCTS root breadth
tracks BatchedMCTS and does not collapse onto a single action.

FAIL-BEFORE / PASS-AFTER (empirically measured, sims=200, K=50, add_noise=False):
  * With the bug:  TensorMCTS breadth ~2.4, single-move share ~90% (FAILS both asserts).
  * With the fix:  TensorMCTS breadth ~9 (≈ BatchedMCTS ~9.7), share spread (PASSES).

Runs entirely on CPU — no CUDA required.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest
import torch

import chess

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.mcts.mcts import BatchedMCTS
from src.mcts.tensor_mcts import TensorMCTS
from src.training.replay_buffer import stack_with_history


def _find_chess_checkpoint() -> str | None:
    """A TRAINED chess checkpoint makes the oracle's value head sane so it
    explores broadly (breadth ~10). A fresh-init net has an arbitrary value
    head that can make the ORACLE itself collapse, which would make this
    regression comparison vacuous — so require a real checkpoint and skip
    otherwise."""
    cands = sorted(
        glob.glob("checkpoints/chess/**/checkpoint_*.pt", recursive=True)
    )
    return cands[-1] if cands else None


# A handful of real, varied middlegame positions (legal-move counts 25-40-ish),
# chosen so the network policy is diffuse enough that the oracle explores
# meaningfully many distinct root actions (breadth ~8-12).
_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pp2pppp/3p1n2/2pP4/4P3/2N5/PPP2PPP/R1BQKBNR b KQkq - 0 4",
    "r2q1rk1/pp1bbppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "2rq1rk1/pb1nbppp/1p2pn2/2pp4/3P4/1P1BPN2/PBPN1PPP/R2Q1RK1 w - - 0 11",
]


def _build_net_and_inputs(ckpt_path: str):
    game = ChessGame()
    cfg = get_config("chess")
    cfg.device = "cpu"
    cfg.num_simulations = 200
    cfg.sample_k = 50

    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Build the net to MATCH the (arbitrary on-disk) checkpoint's architecture by
    # detecting heads from its state_dict keys — same pattern as play_web.load_network.
    # Keeps this test robust to architecture drift (conv policy head, moves-left head,
    # consistency/inverse aux heads) across old and new checkpoints.
    sd = ckpt["model_state_dict"]
    has_conv_policy = any(".policy_head.mix." in k or ".policy_head.proj." in k for k in sd)
    has_moves_left = any(k.startswith("moves_left_head.") for k in sd)
    has_consistency = any(k.startswith("projection.") for k in sd)
    has_inverse = any(k.startswith("inverse_dynamics_head.") for k in sd)

    net = MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size,
        hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks,
        latent_h=cfg.latent_h,
        latent_w=cfg.latent_w,
        input_h=8,
        input_w=8,
        fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size,
        action_embed_dim=cfg.action_embed_dim,
        use_consistency_loss=has_consistency,
        proj_hid=cfg.proj_hid,
        proj_out=cfg.proj_out,
        pred_hid=cfg.pred_hid,
        pred_out=cfg.pred_out,
        use_scalar_transform=cfg.use_scalar_transform,
        value_target_scale=cfg.value_target_scale,
        value_head_type=cfg.value_head_type,
        draw_score=cfg.draw_score,
        value_head_init_std=getattr(cfg, "value_head_init_std", 0.0),
        use_inverse_dynamics_loss=has_inverse,
        inverse_dynamics_hidden=getattr(cfg, "inverse_dynamics_hidden", 256),
        policy_head_type="conv" if has_conv_policy else "flat",
        use_moves_left=has_moves_left,
        moves_left_support_size=getattr(cfg, "moves_left_support_size", 10),
    )
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    obs_list = []
    legal_list = []
    for fen in _FENS:
        st = game.reset()
        st.board = chess.Board(fen)
        # Stack to the configured history depth (zero-padded prior frames) so
        # the observation matches the network's input channel count.
        obs = stack_with_history(game.to_tensor(st), [], cfg.history_frames)
        obs_list.append(obs)
        legal_list.append(game.legal_actions(st))
    return net, game, cfg, obs_list, legal_list


def _root_breadth_and_top_share(root) -> tuple[int, float]:
    """Distinct actions explored (visited slots) and the max single-action share."""
    visits = np.asarray(root.child_visits, dtype=np.float64)
    actions = np.asarray(root.child_actions)
    # Sum visits per unique action (TensorMCTS uses padded -1 slots; the numpy
    # oracle is already deduped, but summing is harmless / correct for both).
    valid = actions != -1
    v = visits[valid]
    a = actions[valid]
    if v.sum() <= 0:
        return 0, 0.0
    per_action: dict[int, float] = {}
    for ai, vi in zip(a.tolist(), v.tolist()):
        per_action[ai] = per_action.get(ai, 0.0) + vi
    counts = np.array(list(per_action.values()))
    breadth = int((counts > 0).sum())
    top_share = float(counts.max() / counts.sum())
    return breadth, top_share


@pytest.mark.parametrize("add_noise", [False])
def test_tensor_mcts_root_breadth_matches_batched(add_noise):
    ckpt = _find_chess_checkpoint()
    if ckpt is None:
        pytest.skip("no trained chess checkpoint found; regression needs a sane "
                    "value head so the oracle explores broadly")
    net, game, cfg, obs_list, legal_list = _build_net_and_inputs(ckpt)

    bm = BatchedMCTS(net, game, cfg, "cpu")
    tm = TensorMCTS(net, game, cfg, device="cpu", compile=False,
                    select_backend="eager")

    bm_breadths, bm_shares = [], []
    tm_breadths, tm_shares = [], []
    for obs, legal in zip(obs_list, legal_list):
        np.random.seed(7)
        torch.manual_seed(7)
        bm._rng = np.random.default_rng(7)
        bm_root = bm.run_batch([obs], [legal], add_noise=add_noise)[0]

        np.random.seed(7)
        torch.manual_seed(7)
        tm_root = tm.run_batch([obs], [legal], add_noise=add_noise)[0]

        bb, bs = _root_breadth_and_top_share(bm_root)
        tb, ts = _root_breadth_and_top_share(tm_root)
        bm_breadths.append(bb); bm_shares.append(bs)
        tm_breadths.append(tb); tm_shares.append(ts)

    bm_mean_breadth = float(np.mean(bm_breadths))
    tm_mean_breadth = float(np.mean(tm_breadths))
    # Per-position EXCESS of TensorMCTS's top-share over the oracle's. The bug
    # signature is TM near-one-hot (~90%) while the oracle spreads (~30%). A
    # legitimately-sharp network peaks on BOTH engines, so compare to the oracle
    # rather than an absolute cap — an absolute 0.85 cap is sensitive to how sharp
    # the loaded `cands[-1]` checkpoint happens to be, which made this test flaky
    # across training progress (it tripped on a sharper checkpoint, passed on others).
    tm_share_excess = float(np.max(np.array(tm_shares) - np.array(bm_shares)))

    # Sanity: the oracle itself must actually be exploring broadly on these
    # positions, otherwise the comparison is vacuous.
    assert bm_mean_breadth >= 6.0, (
        f"oracle (BatchedMCTS) breadth too low ({bm_mean_breadth:.1f}); "
        "positions not diffuse enough to exercise the regression"
    )

    # (1) TensorMCTS must track the oracle's breadth (>= 0.6x) and must NOT be
    #     collapsed to <= 3 distinct actions while the oracle explores ~10.
    assert tm_mean_breadth >= 0.6 * bm_mean_breadth, (
        f"TensorMCTS breadth {tm_mean_breadth:.1f} collapsed vs "
        f"BatchedMCTS {bm_mean_breadth:.1f} (the Option-(b) 1/K-prior bug)"
    )
    assert tm_mean_breadth > 3.0, (
        f"TensorMCTS breadth {tm_mean_breadth:.1f} collapsed to <=3 distinct "
        f"actions while BatchedMCTS explored {bm_mean_breadth:.1f}"
    )

    # (2) TensorMCTS must NOT peak FAR more than the oracle on any position
    #     (the prior-fragmentation signature: TM near-one-hot while BM spreads).
    #     Relative to the oracle so a legitimately-sharp checkpoint doesn't trip it.
    assert tm_share_excess <= 0.40, (
        f"TensorMCTS top-share exceeds the oracle's by {tm_share_excess:.0%} on "
        "some position (TM peaking far more than BatchedMCTS — prior-fragmentation)"
    )
