"""Replay buffer for MuZero training."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


# --- Sparse policy storage (Path B, 2026-06-20) -----------------------------
# MCTS visit-count policies are ~99% zeros (≈30-40 legal/visited moves out of
# 4672). Storing them dense fp32 cost 18.2 KB/ply and dominated the in-RAM
# replay buffer (~10 MB/game -> OOM-kill at the 60 GB cgroup cap once the buffer
# scaled up). We store them sparse as (indices, values) and densify only the
# ~batch_size positions sampled per training step. densify(sparsify(p)) == p
# byte-for-byte, so this is a lossless storage transform — no consumer sees a
# different distribution. `_densify_policy` also passes dense arrays through
# unchanged, so producers that still write dense remain correct (just larger).

def _sparsify_policy(dense) -> tuple:
    """Dense policy array -> (int32 nonzero indices, fp32 values). Lossless."""
    d = np.asarray(dense, dtype=np.float32)
    idx = np.nonzero(d)[0].astype(np.int32)
    return (idx, d[idx].astype(np.float32))


# Standard piece values for the material-margin utility, indexed by the chess
# observation plane order P,N,B,R,Q (planes 0-4 own / 6-10 opponent). King
# (plane 5/11) excluded — both sides always have exactly one, so it cancels.
_MATERIAL_PLANE_VALUES = np.array([1.0, 3.0, 3.0, 5.0, 9.0], dtype=np.float32)


def _material_margin_stm(obs) -> float:
    """STM-relative material margin (in pawns) from a chess observation.

    ``obs`` is a single-frame (C, H, W) tensor/array with the AlphaZero chess
    encoding: planes 0-5 = own pieces (P,N,B,R,Q,K) from the side-to-move POV,
    6-11 = opponent. Returns (own material − opponent material), already
    side-to-move-relative (matches the value-target POV — no parity flip).
    """
    a = obs.numpy() if hasattr(obs, "numpy") else np.asarray(obs)
    own = a[0:5].reshape(5, -1).sum(axis=1)        # P,N,B,R,Q counts (own)
    opp = a[6:11].reshape(5, -1).sum(axis=1)       # P,N,B,R,Q counts (opponent)
    return float(np.dot(_MATERIAL_PLANE_VALUES, own - opp))


def _densify_policy(p, action_space_size):
    """Reconstruct a dense [action_space_size] fp32 policy.

    Accepts a sparse (indices, values) tuple OR a dense ndarray (pass-through
    for back-compat with producers/games that still store dense). None -> None.
    """
    if p is None:
        return None
    if isinstance(p, tuple):
        idx, val = p
        dense = np.zeros(action_space_size, dtype=np.float32)
        if len(idx) > 0:
            dense[np.asarray(idx, dtype=np.int64)] = np.asarray(val, dtype=np.float32)
        return dense
    return p  # already dense — make_target's pad/truncate handles width < A


@dataclass
class GameHistory:
    """Stores a complete game trajectory."""
    observations: list[torch.Tensor] = field(default_factory=list)  # (C, H, W) tensors
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)  # MCTS visit count distributions (fp32)
    root_values: list[float] = field(default_factory=list)  # MCTS root values
    legal_actions_list: list[list[int]] = field(default_factory=list)  # legal actions at each step
    game_outcome: float = 0.0  # final game outcome from player 1's perspective
    game_name: str = ""  # for multi-game training
    # Per-step value targets from an external source (e.g., Stockfish eval), already
    # converted to side-to-move POV in [-1, +1]. Empty for self-play games (default).
    # When populated, make_target prefers these over the td_steps/game_outcome paths.
    external_values: list[float] = field(default_factory=list)
    # No-progress draw flags: True iff this self-play game ended by threefold
    # repetition / by the 75-move (no-progress) rule respectively. Either one
    # makes make_target tilt the (draw) value target toward LOSS (the no-progress
    # shuffle penalty). NOT set for stalemate / insufficient-material / ply-cap.
    draw_by_repetition: bool = False
    draw_by_no_progress: bool = False
    # True iff this game was ended by post-hoc consecutive-move resignation
    # (truncated + relabeled as a decisive loss). Transient self-play stat —
    # used for the self_play/resignation_rate metric; not serialized.
    resigned: bool = False
    # Calibration holdout (AlphaZero-style): True iff this game hit the resign
    # trigger but was left un-resigned (played to its natural end) to measure
    # false positives. resign_false_positive: of those, True iff the would-have-
    # resigned side did NOT actually lose (a wrong resignation we avoided).
    resign_holdout: bool = False
    resign_false_positive: bool = False
    # True iff this game ever reached a ≤tb_max_pieces (tablebase) position while
    # alive. Drives self_play/tb_reach_rate — "how often does the model even reach
    # the stage where root TB probing fires?" Transient, not serialized.
    reached_tb: bool = False
    # True iff this game was truncated at its first decisive in-TB ply and
    # finished with a TB-optimal rollout (tb_rollout_fill): game_outcome is the
    # tablebase verdict and the tail plies are demonstration targets. Exempted
    # from resignation (the demonstration must not be re-truncated). Transient
    # self-play stat (self_play/tb_fill_rate); not serialized.
    tb_filled: bool = False
    # True iff this game's POLICY targets are (partly) tablebase-authored —
    # anchor games (stamped at injection) and rollout-filled games. Reanalyze
    # must NEVER touch these: it would overwrite perfect demonstration targets
    # with the (weaker-in-endgame) net's own MCTS visits. SERIALIZED, so the
    # protection survives buffer save/load.
    tb_authored: bool = False
    # SAMPLING CHANNEL TAG (task #19, unified buffer): "" for organic games;
    # "anchor" stamped at TB-anchor injection. NOT derivable from tb_authored
    # alone (rollout-FILLED self-play games also set tb_authored for the
    # reanalyze guard, but belong to the selfplay channel). Serialized.
    channel_tag: str = ""
    # Per-ply STM-relative Syzygy value of the POSITION (DTZ-shaped), in [-1, 1],
    # NaN where the ply was not in TB range. Empty for games that never reached
    # the tablebase. When populated, make_target blends it into the VALUE target
    # at TB plies (tb_value_weight) — Lc0-rescorer-style value relabeling that
    # gives the value head a distance-to-mate gradient the flat outcome one-hot
    # lacks. Serialized (NaN survives float32 round-trip).
    tablebase_values: list[float] = field(default_factory=list)
    # Per-ply |DTM| (plies-to-mate, Gaviota), NaN where the ply was not in TB range
    # or the position is drawn. Empty for games that never reached the tablebase.
    # When populated, sample_batch substitutes it for the policy-rollout plies-to-end
    # MOVES-LEFT target at TB plies (tb_moves_left_weight) — Lc0 Gaviota MLH rescoring,
    # the distance gradient the WDL value head provably can't supply. Serialized.
    tablebase_moves_left: list[float] = field(default_factory=list)
    # Per-ply SOFT TB policy target (idx int32[], weight float32[]) over the
    # win-PRESERVING moves, or None where the ply was not in TB range / had no winning
    # move. When populated, sample_batch blends it into the policy target at TB plies
    # (tb_policy_weight) — the Lc0 DTZ policy boost, safe for us because we don't run
    # KLDGain (see kld_adaptive_search_analysis). Annealed. Fixes the policy prior that
    # mass-loads the win-throwing move (move-selection diagnosis 2026-06-27).
    tablebase_policy: list = field(default_factory=list)
    # Optional starting position (FEN) for games that do NOT begin from the
    # standard initial position — e.g. endgame-seed curriculum games. When set,
    # from_compact_dict replays actions from this FEN instead of game.reset().
    # None (default) => standard start; existing shards omit the key entirely.
    start_fen: str | None = None
    # Number of times this game has been reanalyzed (policies/root_values
    # overwritten by the current net's MCTS). 0 => never reanalyzed, so the
    # stored policy targets still match the net-at-generation-time. Set
    # deterministically by Trainer._reanalyze. Warmstart games stay 0 (excluded
    # from reanalyze). Lets the buffer viewer state for sure whether a game's
    # policy targets were recomputed (decoupled from the played moves).
    reanalyze_count: int = 0
    # Monotonic insertion counter, stamped by ReplayBuffer.save_game and used by
    # the buffer's retention-weighted eviction (decisive games age slower).
    # Runtime-only — not serialized in the compact dict; reassigned on load().
    birth: int = 0

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def channel(self) -> str:
        """Sampling channel (task #19): 'warmstart' (Stockfish-eval games),
        'anchor' (injected TB demonstrations), or 'selfplay' (everything
        organic, including rollout-filled games). Derived from existing
        markers + channel_tag so legacy buffers need no migration (legacy
        anchor games without the tag classify as selfplay — acceptable, they
        only exist in pre-task-19 snapshots)."""
        if self.external_values:
            return "warmstart"
        if self.channel_tag == "anchor":
            return "anchor"
        return "selfplay"

    def to_compact_dict(self) -> dict:
        """Pack into a minimal dict for storage: drops observations and
        legal_actions_list (reconstructable by replaying actions through the
        game), and packs policies sparsely.

        One-hot policies (Stockfish shards) collapse to a single int per step.
        Soft policies (self-play) keep per-step (nonzero_indices, values)
        pairs — typically ~30-40 nonzeros for chess vs 4672 dense entries.

        Callers round-trip via ``GameHistory.from_compact_dict(d, game)``.
        """
        n = len(self.actions)

        # Normalize each policy to (indices, values) — handles both the sparse
        # in-RAM tuple form (Path B) and legacy dense ndarrays.
        def _to_idx_val(p):
            if p is None:
                return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32))
            if isinstance(p, tuple):
                idx, val = p
                return (np.asarray(idx, dtype=np.int32), np.asarray(val, dtype=np.float32))
            d = np.asarray(p, dtype=np.float32)
            nz = np.nonzero(d)[0].astype(np.int32)
            return (nz, d[nz].astype(np.float32))

        idx_val = [_to_idx_val(p) for p in self.policies]

        one_hot = (
            len(self.policies) == n
            and all(
                len(idx) == 1 and float(val[0]) > 0.999
                for idx, val in idx_val
            )
        )

        if one_hot:
            policies_mode = "onehot"
            policies_data = np.fromiter(
                (int(idx[0]) for idx, _ in idx_val),
                dtype=np.int32,
                count=n,
            )
        else:
            policies_mode = "sparse"
            policies_data = list(idx_val)

        d = {
            "format_version": 3,
            "game_name": self.game_name,
            "actions": np.asarray(self.actions, dtype=np.int32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "root_values": np.asarray(self.root_values, dtype=np.float32),
            "game_outcome": float(self.game_outcome),
            "policies_mode": policies_mode,
            "policies_data": policies_data,
            "draw_by_repetition": bool(self.draw_by_repetition),
            "draw_by_no_progress": bool(self.draw_by_no_progress),
        }
        if self.external_values:
            d["external_values"] = np.asarray(self.external_values, dtype=np.float32)
        if self.tablebase_values:
            d["tablebase_values"] = np.asarray(self.tablebase_values, dtype=np.float32)
        if self.tablebase_moves_left:
            d["tablebase_moves_left"] = np.asarray(self.tablebase_moves_left, dtype=np.float32)
        if self.tablebase_policy and any(p is not None for p in self.tablebase_policy):
            # Flat encoding of the per-ply sparse soft policy: lengths[ply]=0 => None.
            lengths, flat_idx, flat_w = [], [], []
            for p in self.tablebase_policy:
                if p is None:
                    lengths.append(0)
                else:
                    idx, w = p
                    lengths.append(len(idx))
                    flat_idx.append(np.asarray(idx, dtype=np.int32))
                    flat_w.append(np.asarray(w, dtype=np.float32))
            d["tbp_lengths"] = np.asarray(lengths, dtype=np.int32)
            d["tbp_idx"] = (np.concatenate(flat_idx) if flat_idx else np.zeros(0, dtype=np.int32))
            d["tbp_w"] = (np.concatenate(flat_w) if flat_w else np.zeros(0, dtype=np.float32))
        if self.start_fen:
            d["start_fen"] = self.start_fen
        if self.reanalyze_count:
            d["reanalyze_count"] = int(self.reanalyze_count)
        if self.tb_authored:
            d["tb_authored"] = True
        if self.channel_tag:
            d["channel_tag"] = self.channel_tag
        return d

    @classmethod
    def from_compact_dict(cls, d: dict, game) -> "GameHistory":
        """Reconstruct a GameHistory by replaying ``d['actions']`` through ``game``.

        Rebuilds observations + legal_actions_list; expands sparse policies back
        to dense fp32 arrays of width ``game.action_space_size`` (what
        ``make_target`` expects). ``game`` must be a fresh instance whose
        ``reset()`` returns the same starting state as the original game.
        """
        version = d.get("format_version")
        if version != 3:
            raise ValueError(f"Unknown compact format version: {version!r}")

        gh = cls(game_name=d.get("game_name", ""))
        gh.actions = [int(a) for a in d["actions"]]
        gh.rewards = [float(r) for r in d["rewards"]]
        gh.root_values = [float(v) for v in d["root_values"]]
        gh.game_outcome = float(d["game_outcome"])
        ev = d.get("external_values")
        if ev is not None and len(ev) > 0:
            gh.external_values = [float(v) for v in ev]
        tbv = d.get("tablebase_values")
        if tbv is not None and len(tbv) > 0:
            gh.tablebase_values = [float(v) for v in tbv]
        tbm = d.get("tablebase_moves_left")
        if tbm is not None and len(tbm) > 0:
            gh.tablebase_moves_left = [float(v) for v in tbm]
        tbp_lengths = d.get("tbp_lengths")
        if tbp_lengths is not None and len(tbp_lengths) > 0:
            tbp_idx = np.asarray(d["tbp_idx"], dtype=np.int32)
            tbp_w = np.asarray(d["tbp_w"], dtype=np.float32)
            pol, off = [], 0
            for L in tbp_lengths:
                L = int(L)
                if L == 0:
                    pol.append(None)
                else:
                    pol.append((tbp_idx[off:off + L].copy(), tbp_w[off:off + L].copy()))
                    off += L
            gh.tablebase_policy = pol
        gh.draw_by_repetition = bool(d.get("draw_by_repetition", False))
        gh.draw_by_no_progress = bool(d.get("draw_by_no_progress", False))

        gh.reanalyze_count = int(d.get("reanalyze_count", 0))
        gh.tb_authored = bool(d.get("tb_authored", False))
        gh.channel_tag = str(d.get("channel_tag", ""))
        start_fen = d.get("start_fen")
        gh.start_fen = start_fen
        if start_fen is not None and hasattr(game, "reset_from_fen"):
            state = game.reset_from_fen(start_fen)
        else:
            state = game.reset()
        gh.observations.append(game.to_tensor(state))
        for a in gh.actions:
            gh.legal_actions_list.append(game.legal_actions(state))
            state, _, _ = game.step(state, a)
            gh.observations.append(game.to_tensor(state))

        mode = d["policies_mode"]
        if mode == "onehot":
            # Each ply -> sparse single-entry policy (argmax, prob 1.0). Kept
            # sparse in RAM (Path B); make_target densifies on read.
            policies = [
                (np.asarray([int(idx)], dtype=np.int32),
                 np.asarray([1.0], dtype=np.float32))
                for idx in d["policies_data"]
            ]
            gh.policies = policies
        elif mode == "sparse":
            # Keep the on-disk (indices, values) sparse form in RAM as-is.
            gh.policies = [
                (np.asarray(idxs, dtype=np.int32), np.asarray(vals, dtype=np.float32))
                for idxs, vals in d["policies_data"]
            ]
        else:
            raise ValueError(f"Unknown policies_mode: {mode!r}")

        return gh

    def _stack_history(self, idx: int, n_frames: int):
        """Stack ``n_frames`` consecutive observations ending at ``idx``,
        newest-first along the channel dim. Zero-pad for missing earlier
        frames (idx < n_frames - 1, i.e. early game).

        Returns a torch.Tensor of shape (C * n_frames, H, W) where C is the
        per-frame channel count. For n_frames == 1, this reduces to a single
        observation (current behavior).
        """
        if n_frames <= 1:
            return self.observations[min(idx, len(self.observations) - 1)]
        cap = min(idx, len(self.observations) - 1)
        ref = self.observations[cap]  # for shape / dtype / device
        zero = None
        frames = []
        for t in range(n_frames):
            i = idx - t
            if 0 <= i < len(self.observations):
                frames.append(self.observations[i])
            else:
                if zero is None:
                    zero = torch.zeros_like(ref)
                frames.append(zero)
        # newest first along channel dim
        return torch.cat(frames, dim=0)

    def make_target(self, state_index: int, num_unroll_steps: int,
                    td_steps: int, discount: float, action_space_size: int,
                    value_head_type: str = "support",
                    history_frames: int = 1,
                    eval_to_wdl_alpha: float = 4.0,
                    eval_to_wdl_beta: float = 2.0,
                    q_ratio: float = 0.0,
                    warmstart_q_ratio: float | None = None,
                    selfplay_q_ratio: float | None = None,
                    repetition_penalty: float = 0.0,
                    repetition_penalty_window: int = 0,
                    repetition_penalty_decay: float = 0.0,
                    material_value_weight: float = 0.0,
                    material_value_scale: float = 5.0,
                    tb_value_weight: float = 0.0,
                    tb_policy_weight: float = 0.0,
                    tb_value_hard: bool = False):
        """Create training target for a given position.

        Args:
            value_head_type: 'support' (default, scalar value targets via
                bootstrap or external_values) or 'wdl' (3-vector one-hot WDL
                targets from game outcome — Lc0 style; td_steps and
                external_values are ignored under WDL since the target is
                always the actual game result from each ply's STM POV).

        Returns:
            target_observations: list of K+1 obs at state_index..state_index+K
                (past game end: last valid obs is reused; the mask zeros the loss)
            target_obs_mask: list of K+1 floats — 1.0 where the index is within
                the game, 0.0 past game end. Used by EfficientZero consistency loss.
            target_actions: K future actions (padded with 0 if game ended)
            target_values: K+1 entries — scalars under 'support', shape-(3,)
                np.ndarray under 'wdl'.
            target_rewards: K+1 rewards
            target_policies: K+1 policy targets
        """
        values = []
        rewards = []
        policies = []
        actions = []
        obs_list = []
        obs_mask = []
        last_obs = self._stack_history(min(state_index, len(self) - 1), history_frames)

        # WDL targets: cache draw vector + outcome lookup once per call.
        wdl_draw = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        wdl_w = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        wdl_l = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Lazy import to avoid model→buffer import cycle.
        from src.model.utils import eval_to_wdl as _eval_to_wdl

        def _outcome_onehot(ply_idx: int) -> np.ndarray:
            """STM-relative one-hot of the actual game outcome at ply_idx."""
            if self.game_outcome == 0.0:
                return wdl_draw
            stm_is_white = (ply_idx % 2 == 0)
            stm_won = (self.game_outcome > 0.0) == stm_is_white
            return wdl_w if stm_won else wdl_l

        # Separate blend weights for the two phases. Warmstart blends in the
        # GAME OUTCOME against an EXTERNAL teacher eval (safe — not self-
        # referential), so it can run hot; self-play blends in the network's
        # OWN MCTS root value (self-referential — a feedback loop risk), so it
        # should run cool, matching AlphaZero/Lc0's near-zero default. Both fall
        # back to the single ``q_ratio`` when not explicitly split.
        q_warm = float(warmstart_q_ratio if warmstart_q_ratio is not None else q_ratio)
        q_self = float(selfplay_q_ratio if selfplay_q_ratio is not None else q_ratio)
        # No-progress shuffle penalty (self-play only). When a self-play game drew
        # by threefold repetition OR by the 75-move no-progress rule, move δ mass
        # from Draw → Loss in the legacy outcome target: [0, 1, 0] → [0, 1-δ, δ].
        # STM-symmetric (same for both sides). NOT applied to stalemate /
        # insufficient-material / ply-cap draws. δ=0 reproduces prior behavior.
        rep_delta = float(min(max(repetition_penalty, 0.0), 1.0))
        rep_window = int(max(repetition_penalty_window, 0))
        rep_decay = float(min(max(repetition_penalty_decay, 0.0), 1.0))
        # Final ply index (terminal/drawn position is at len-1) for per-ply ramp.
        rep_end_idx = len(self) - 1
        # Material/score-margin utility (chess self-play only). Guarded to chess
        # because the piece-plane layout (planes 0-4/6-10) is chess-specific.
        mat_w = (float(min(max(material_value_weight, 0.0), 1.0))
                 if self.game_name == "chess" else 0.0)
        mat_scale = float(material_value_scale) if material_value_scale > 0.0 else 5.0
        # Tablebase value-relabel weight (gated purely on tablebase_values being
        # populated — only chess self-play games carry it). 1.0 = full Syzygy
        # replacement at TB plies; only applies on the WDL self-play path below.
        tb_w = float(min(max(tb_value_weight, 0.0), 1.0)) if self.tablebase_values else 0.0

        def _wdl_target_at(ply_idx: int) -> np.ndarray:
            """WDL target at ply_idx from side-to-move's POV, with q_ratio blend.

            q (q_warm or q_self) weights the "blended-in" signal; (1-q) weights
            the phase's legacy target. Blending two probability distributions
            with weights summing to 1 yields a valid distribution.

            Two phases:
            (1) Warmstart games (external_values present at this ply): the legacy
                target is the soft WDL from Stockfish's per-position eval via
                eval_to_wdl(); q_warm blends in the GAME OUTCOME one-hot.
                    target = (1-q_warm)·eval_to_wdl(external) + q_warm·outcome_onehot
            (2) Self-play games (no external_values, or ply past their end): the
                legacy target is the STM-relative outcome one-hot (Lc0 pure-z);
                q_self blends in the MCTS ROOT VALUE mapped to WDL via eval_to_wdl().
                    target = (1-q_self)·outcome_onehot + q_self·eval_to_wdl(root_value)
                If root_values is missing/short for this ply, fall back to pure
                outcome_onehot (q effectively 0).

            At q == 0.0 both phases reduce exactly to the legacy target.
            """
            # Per-position eval available → warmstart phase.
            if self.external_values and ply_idx < len(self.external_values):
                q = q_warm
                stm_eval = float(self.external_values[ply_idx])
                # external_values is already side-to-move-relative (per the
                # generate_stockfish_games convention) so no parity flip needed.
                p_w, p_d, p_l = _eval_to_wdl(
                    stm_eval, alpha=eval_to_wdl_alpha, beta=eval_to_wdl_beta
                )
                legacy = np.array([p_w, p_d, p_l], dtype=np.float32)
                if q == 0.0:
                    return legacy
                blended_in = _outcome_onehot(ply_idx)
                target = (1.0 - q) * legacy + q * blended_in
                assert abs(float(target.sum()) - 1.0) < 1e-5, target
                return target.astype(np.float32)
            # Self-play phase: legacy target is the outcome one-hot.
            q = q_self
            legacy = _outcome_onehot(ply_idx)
            # No-progress shuffle penalty: tilt a draw target toward LOSS before
            # blending. Only for no-progress draws (threefold OR 75-move, with
            # game_outcome == 0.0); decisive games, stalemate, insufficient
            # material, and ply-cap draws are untouched.
            # STM-winning guard (2026-06-19 bug hunt): the tilt was STM-SYMMETRIC and
            # taught the objectively-winning side of a won-but-shuffled position that it
            # was LOSING. Skip it when the stored root value says STM is clearly winning
            # (rv > guard); in the draw basin (rv≈0) the tilt still applies as before.
            _rv_here = (float(self.root_values[ply_idx])
                        if ply_idx < len(self.root_values) else 0.0)
            stm_winning = _rv_here > 0.25
            if (rep_delta > 0.0 and self.game_outcome == 0.0
                    and (self.draw_by_repetition or self.draw_by_no_progress)
                    and not stm_winning):
                # Per-ply shuffle-depth weighting: full δ at the drawn position,
                # decaying backward. Exponential (decay**plies_to_end, preferred —
                # discount-style soft tail) takes precedence over the linear window;
                # both off ⇒ weight 1 everywhere (uniform/legacy).
                if rep_decay > 0.0:
                    plies_to_end = rep_end_idx - ply_idx
                    weight = rep_decay ** plies_to_end
                elif rep_window > 0:
                    plies_to_end = rep_end_idx - ply_idx
                    weight = max(0.0, 1.0 - plies_to_end / float(rep_window))
                else:
                    weight = 1.0
                d = rep_delta * weight
                legacy = np.array([0.0, 1.0 - d, d], dtype=np.float32)
            # Material/score-margin blend (KataGo c_score analogue): make two
            # winning positions rank-able so the value head gets a within-
            # position gradient even on drawn-by-shuffle games (outcome=draw but
            # material=+rook). Applied to the legacy (outcome) target BEFORE the
            # q_self root-value blend, self-play phase only.
            if mat_w > 0.0 and ply_idx < len(self.observations):
                m_eval = float(np.tanh(
                    _material_margin_stm(self.observations[ply_idx]) / mat_scale))
                p_w, p_d, p_l = _eval_to_wdl(
                    m_eval, alpha=eval_to_wdl_alpha, beta=eval_to_wdl_beta)
                material_wdl = np.array([p_w, p_d, p_l], dtype=np.float32)
                legacy = (1.0 - mat_w) * legacy + mat_w * material_wdl
            # Tablebase value relabeling (Lc0-rescorer analogue). At plies inside
            # the Syzygy tablebase, blend the DTZ-shaped position value (an
            # external oracle → feedback-safe, like the warmstart Stockfish
            # teacher) into the value target. Gives the value head the distance-
            # to-mate gradient the flat outcome one-hot lacks; for won-but-shuffled
            # draws it correctly relabels the draw target toward a win. Already
            # STM-relative → no parity flip. Applied AFTER the material blend and
            # BEFORE the q_self blend so a self-referential root-value blend can't
            # override the oracle (keep selfplay_q_ratio≈0 for TB dominance).
            if (tb_w > 0.0 and ply_idx < len(self.tablebase_values)):
                tv = self.tablebase_values[ply_idx]
                if tv == tv:  # not NaN
                    if tb_value_hard:
                        # Lc0-style HARD WDL: the tablebase outcome is certain, so
                        # train the value head on a one-hot (win/draw/loss) instead of
                        # softening it through eval_to_wdl (which caps W-L at ~0.88 and
                        # never saturates). Saturates Q near ±1 → crisp win/draw
                        # separation (refutes throws) + activates the |Q|-gated MLH,
                        # which then carries the within-win distance. tv is the STM-POV
                        # TB scalar (±1 / 0, or ±dtz-shaped if dtz_shape>0); threshold on
                        # sign. Draw / cursed-win / blessed-loss → draw one-hot.
                        if tv > 0.5:
                            tb_wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                        elif tv < -0.5:
                            tb_wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                        else:
                            tb_wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    else:
                        p_w, p_d, p_l = _eval_to_wdl(
                            float(tv), alpha=eval_to_wdl_alpha, beta=eval_to_wdl_beta)
                        tb_wdl = np.array([p_w, p_d, p_l], dtype=np.float32)
                    legacy = (1.0 - tb_w) * legacy + tb_w * tb_wdl
            if q == 0.0:
                return legacy.astype(np.float32)
            # Blend in the MCTS root value (STM POV scalar) mapped to WDL.
            if ply_idx < len(self.root_values):
                rv = float(self.root_values[ply_idx])
                p_w, p_d, p_l = _eval_to_wdl(
                    rv, alpha=eval_to_wdl_alpha, beta=eval_to_wdl_beta
                )
                blended_in = np.array([p_w, p_d, p_l], dtype=np.float32)
                target = (1.0 - q) * legacy + q * blended_in
                assert abs(float(target.sum()) - 1.0) < 1e-5, target
                return target.astype(np.float32)
            # root_values missing/short → fall back to pure outcome (q≈0).
            return legacy

        for i in range(num_unroll_steps + 1):
            idx = state_index + i

            if idx < len(self):
                if value_head_type == "wdl":
                    # Two paths inside _wdl_target_at:
                    #  - Warmstart games (external_values present) → soft WDL
                    #    derived from Stockfish per-position eval. Rich signal.
                    #  - Self-play games (no external_values) → one-hot of
                    #    game outcome from STM POV. Lc0 pure-z target.
                    value = _wdl_target_at(idx)
                elif self.external_values and idx < len(self.external_values):
                    # Warmstart path: external targets are already side-to-move-relative.
                    value = self.external_values[idx]
                elif td_steps == -1:
                    sign = 1.0 if (idx % 2 == 0) else -1.0
                    value = sign * self.game_outcome
                else:
                    bootstrap_idx = idx + td_steps
                    value = 0.0
                    for j in range(idx, min(bootstrap_idx, len(self))):
                        r = self.rewards[j] if j < len(self.rewards) else 0.0
                        # 2-player zero-sum: self.rewards[j] is in the mover-at-j
                        # POV; rebase to the STM-at-idx POV by flipping when the
                        # plies have opposite parity.
                        parity = 1.0 if ((j % 2) == (idx % 2)) else -1.0
                        value += (discount ** (j - idx)) * parity * r
                    # root_values has one fewer entry than observations (no root_value
                    # for the terminal observation), so guard against that off-by-one.
                    if bootstrap_idx < len(self.root_values):
                        sign = 1.0 if ((bootstrap_idx % 2) == (idx % 2)) else -1.0
                        value += (discount ** td_steps) * sign * self.root_values[bootstrap_idx]

                values.append(value)
                # DeepMind MuZero convention: target_rewards[i] is the reward of the
                # transition INTO state (state_index + i), i.e. self.rewards[idx - 1].
                # target_rewards[0] is therefore the reward INTO the root and is unused
                # by the trainer (initial_inference predicts no reward). The trainer
                # reads target_rewards[:, k+1] at unroll step k, which under this
                # indexing equals self.rewards[state_index + k] — the reward of
                # transition s_{state_index+k} → s_{state_index+k+1}, which is exactly
                # what recurrent_inference(hidden_k, actions[:, k]) predicts.
                prev = idx - 1
                rewards.append(self.rewards[prev] if 0 <= prev < len(self.rewards) else 0.0)

                policy = (_densify_policy(self.policies[idx], action_space_size)
                          if idx < len(self.policies) else None)
                if policy is None or len(policy) == 0:
                    # INTENTIONAL benign no-op (bug_hunt_2026_06_13.md §E).
                    # Root sampling (sample_batch: randint(0, len(game))) can land
                    # on the terminal index, where idx == len(actions) so there is
                    # no MCTS policy (self.policies has one entry per non-terminal
                    # ply). A ZEROS target — unlike a uniform target — contributes
                    # exactly 0 CE gradient, so this position trains its (correct,
                    # nonzero, STM-POV) VALUE target only and pulls the policy head
                    # toward nothing. Do NOT change this to uniform/argmax: that
                    # would inject a spurious policy gradient at terminal states.
                    policy = np.zeros(action_space_size, dtype=np.float32)
                else:
                    policy = np.asarray(policy, dtype=np.float32)
                    if len(policy) < action_space_size:
                        padded = np.zeros(action_space_size, dtype=np.float32)
                        padded[: len(policy)] = policy
                        policy = padded
                    elif len(policy) > action_space_size:
                        policy = policy[:action_space_size]
                # Soft TB policy relabel (Lc0 DTZ boost, safe sans KLDGain): blend the
                # win-preserving distribution into the visit policy at TB plies —
                # (1-w)*visits + w*tb. Only where a real visit policy exists (skip the
                # terminal-zeros case so we inject no spurious gradient).
                if (tb_policy_weight > 0.0 and self.tablebase_policy
                        and idx < len(self.tablebase_policy)
                        and self.tablebase_policy[idx] is not None
                        and float(policy.sum()) > 0.0):
                    tbp_idx, tbp_w = self.tablebase_policy[idx]
                    tbp_idx = np.asarray(tbp_idx, dtype=np.int64)
                    valid = tbp_idx < action_space_size
                    if valid.any():
                        tb_dense = np.zeros(action_space_size, dtype=np.float32)
                        tb_dense[tbp_idx[valid]] = np.asarray(tbp_w, dtype=np.float32)[valid]
                        s = float(tb_dense.sum())
                        if s > 0.0:
                            tb_dense /= s
                            policy = ((1.0 - tb_policy_weight) * policy
                                      + tb_policy_weight * tb_dense).astype(np.float32)
                policies.append(policy)
                last_obs = self._stack_history(idx, history_frames)
                obs_list.append(last_obs)
                obs_mask.append(1.0)
            else:
                # Past game end: the obs_mask zeros these out of the loss,
                # so the precise filler value doesn't matter for training.
                # Use the matching shape so downstream tensor stacking works.
                if value_head_type == "wdl":
                    values.append(wdl_draw)
                else:
                    values.append(0.0)
                rewards.append(0.0)
                policies.append(np.full(action_space_size, 1.0 / action_space_size, dtype=np.float32))
                obs_list.append(last_obs)
                obs_mask.append(0.0)

            if i < num_unroll_steps:
                act_idx = state_index + i
                if act_idx < len(self.actions):
                    actions.append(self.actions[act_idx])
                else:
                    actions.append(0)

        return (
            obs_list,
            obs_mask,
            actions,
            values,
            rewards,
            policies,
        )


def stack_with_history(current_obs, prior_obs, n_frames: int):
    """Build a T-frame stacked observation for inference time.

    ``current_obs``: the current ply's observation (single-frame, shape (C, H, W)).
    ``prior_obs``: chronologically-ordered list of earlier-ply observations,
                   newest at the end (i.e. ``prior_obs[-1]`` is one ply before
                   ``current_obs``).
    ``n_frames``: total stack depth T.

    Returns a tensor of shape (T*C, H, W) with newest frame at channel 0.
    Zero-pads for missing earlier frames (early-game). Same plane order as
    ``GameHistory._stack_history`` (used at training sample time).

    For ``n_frames <= 1`` returns ``current_obs`` unchanged — no allocation,
    so callers can use this regardless of history config.
    """
    if n_frames <= 1:
        return current_obs
    frames = [current_obs]
    for t in range(1, n_frames):
        i = len(prior_obs) - t  # t=1 → most recent prior
        if 0 <= i < len(prior_obs):
            frames.append(prior_obs[i])
        else:
            frames.append(torch.zeros_like(current_obs))
    return torch.cat(frames, dim=0)


def _iter_shard_games(path: str | Path, game=None):
    """Yield GameHistory objects from a Stockfish shard, legacy or compact.

    Legacy shard: a single pickled ``list[GameHistory]``.
    Compact (v2) shard: a header dict followed by N compact-dict pickles;
    each record is decoded via ``GameHistory.from_compact_dict(d, game)``.

    Used by both ``ReplayBuffer.load_warmstart_games`` and the trainer's
    per-shard injection loader.
    """
    with open(path, "rb") as f:
        first = pickle.load(f)
        if isinstance(first, list):
            for g in first:
                # Canonical shards (generate_stockfish_games.py --format-version=1) carry
                # observations as numpy arrays; rewrap to torch.Tensor (zero-copy view) so
                # downstream code is unchanged. Storing torch.Tensor in v1 pickles SEGVs
                # training via torch's tensor storage reducer — see CRASH_INVESTIGATION.md.
                if g.observations and isinstance(g.observations[0], np.ndarray):
                    g.observations = [torch.from_numpy(o) for o in g.observations]
                yield g
        elif isinstance(first, dict) and first.get("version") == 2:
            if game is None:
                raise ValueError(
                    f"{path}: compact (v2) shard requires `game` for reconstruction"
                )
            for _ in range(first["n_records"]):
                d = pickle.load(f)
                yield GameHistory.from_compact_dict(d, game)
        else:
            raise TypeError(
                f"{path}: unrecognized shard format "
                f"(got {type(first).__name__}, expected list or version-2 header dict)"
            )


def _shard_record_count(path: "str | Path") -> int:
    """Number of games in a shard, read CHEAPLY — no per-game reconstruction.

    Used by injection fast-forward on resume to skip already-consumed shards
    without replaying every game's moves through the engine (the cause of the
    resume hang: reconstructing ~19k games just to discard them).

    v2 compact shard: returns only the header's ``n_records`` (one pickle read).
    Legacy list shard: loads the list (no move-replay) and returns its length.
    """
    with open(path, "rb") as f:
        first = pickle.load(f)
        if isinstance(first, list):
            return len(first)
        if isinstance(first, dict) and first.get("version") == 2:
            return int(first["n_records"])
        raise TypeError(
            f"{path}: unrecognized shard format "
            f"(got {type(first).__name__}, expected list or version-2 header dict)"
        )


class ReplayBuffer:
    """Fixed-size replay buffer with Prioritized Experience Replay (PER).

    Games are sampled proportional to priority^alpha. Importance sampling
    weights correct for the sampling bias introduced by prioritization.
    Priorities are updated after each training step using TD errors.

    With alpha=0 or beta=1, degrades gracefully to uniform sampling.
    """

    def __init__(self, max_size: int, warmstart_max_size: int | None = None,
                 decisive_retention_multiplier: float = 1.0,
                 anchor_max_size: int | None = None):
        self.max_size = max_size
        # Per-channel cap for injected TB-anchor demonstrations (task #19).
        # None = legacy behavior (anchor games compete in the selfplay pool,
        # so anchor share is EMERGENT from injection cadence vs eviction).
        # Set => anchor becomes a third pool with its own within-pool eviction,
        # making demonstration volume a chosen number.
        self.anchor_max_size = anchor_max_size
        # Retention "lives": when > 1, a DECISIVE game's age counts this many times
        # slower for eviction, so it survives ~M× longer than a drawn game (lifting
        # the buffer's decisive-game density). 1.0 = plain oldest-first FIFO.
        # Steady-state decisive buffer fraction with decisive inflow d:
        #   d*M / (d*M + (1-d))   → at d=0.05: M=3→14%, M=7→27%, M=14→42%.
        self.decisive_retention_multiplier = float(decisive_retention_multiplier)
        # Monotonic counter stamped onto each game's `birth` for age-based eviction.
        self._save_counter = 0
        # Two-pool FIFO mode. When ``warmstart_max_size`` is set, the buffer is
        # logically partitioned by ``external_values``: warmstart games (set)
        # vs self-play games (unset). Each pool evicts independently within its
        # own cap, so warmstart games can never be FIFO'd out by incoming
        # self-play traffic. Required for ``warmstart_sample_frac > 0`` to keep
        # working past the warmstart-pool exhaustion phase. Without it,
        # warmstart games age out within ~10 self-play rounds and the
        # stratified sampler silently degrades to flat sampling.
        self.warmstart_max_size = warmstart_max_size
        self.buffer: list[GameHistory] = []
        self._priorities: list[float] = []
        self.total_games = 0

    def _eviction_victim(self, is_warm: bool | None, channel: str | None = None) -> int | None:
        """Index of the game to evict among the candidate pool.

        Evicts the game with the largest ``age / retention_weight`` — i.e. the one
        that has overstayed its allowance the most. ``age`` = saves since insertion;
        ``weight`` = decisive_retention_multiplier for DECISIVE games, 1 for draws.
        So decisive games are evicted only once they're ~M× older than a draw would
        be → they persist ~M× longer. With multiplier 1.0 this is plain oldest-first.

        ``channel`` (task #19) selects the candidate pool by GameHistory.channel;
        else ``is_warm`` selects the legacy two-pool split; both None = single pool.
        """
        M = self.decisive_retention_multiplier
        best_i, best_score = None, -1.0
        for i, g in enumerate(self.buffer):
            if channel is not None:
                if g.channel != channel:
                    continue
            elif is_warm is not None and bool(g.external_values) != is_warm:
                continue
            age = self._save_counter - g.birth
            weight = M if (M > 1.0 and g.game_outcome != 0.0) else 1.0
            score = age / weight
            if score > best_score:
                best_score, best_i = score, i
        return best_i

    def save_game(self, game_history: GameHistory):
        # New games get max current priority so they're sampled at least once
        max_p = max(self._priorities) if self._priorities else 1.0
        game_history.birth = self._save_counter
        self._save_counter += 1
        is_warm = bool(game_history.external_values)
        if self.anchor_max_size is not None:
            # Three-pool eviction (task #19): each channel evicts only among
            # itself once its cap is reached — every pool's volume is a CHOSEN
            # number (warm = warmstart_max_size, anchor = anchor_max_size,
            # selfplay = the remainder). Victims retention-weighted as before.
            ch = game_history.channel
            # warmstart_max_size=None means UNBOUNDED — warmstart fills the whole
            # non-anchor remainder. Matches the two-pool branch's treatment of None
            # and is what the warmup two-phase sizing relies on (trainer sets
            # warmstart_max_size=None so warmstart fills the buffer before the
            # self-play boundary trims it to warmstart_buffer_size). Without this,
            # `None or 0` capped warmstart at 0 and every injected game was evicted.
            if self.warmstart_max_size is not None:
                warm_cap = self.warmstart_max_size
            else:
                warm_cap = self.max_size - self.anchor_max_size
            caps = {
                "warmstart": warm_cap,
                "anchor": self.anchor_max_size,
                "selfplay": max(1, self.max_size - warm_cap - self.anchor_max_size),
            }
            cur = sum(1 for g in self.buffer if g.channel == ch)
            if cur >= caps[ch]:
                victim = self._eviction_victim(None, channel=ch)
                if victim is not None:
                    self.buffer.pop(victim)
                    self._priorities.pop(victim)
        elif self.warmstart_max_size is not None:
            # Two-pool eviction: if the relevant pool is at cap, evict within THAT
            # pool only (warmstart games can never be displaced by self-play). The
            # victim is retention-weighted (decisive games persist M× longer).
            sp_max = self.max_size - self.warmstart_max_size
            cap = self.warmstart_max_size if is_warm else sp_max
            cur = sum(
                1 for g in self.buffer
                if bool(g.external_values) == is_warm
            )
            if cur >= cap:
                victim = self._eviction_victim(is_warm)
                if victim is not None:
                    self.buffer.pop(victim)
                    self._priorities.pop(victim)
        else:
            # Legacy single-pool eviction (retention-weighted; M=1 ⇒ FIFO).
            if len(self.buffer) >= self.max_size:
                victim = self._eviction_victim(None)
                if victim is not None:
                    self.buffer.pop(victim)
                    self._priorities.pop(victim)
        self.buffer.append(game_history)
        self._priorities.append(max_p)
        self.total_games += 1

    def trim_warmstart_to(self, n: int) -> int:
        """Evict oldest warmstart games until at most ``n`` remain.

        Used at the two-phase boundary: Phase 1 fills the buffer with warmstart
        games (no self-play to crowd out → no anchor cap needed), then this
        trims the warmstart pool down to the persistent anchor size right before
        self-play begins — otherwise the buffer would balloon past ``max_size``
        once self-play games start arriving (two-pool eviction never drops
        warmstart for self-play). Keeps the ``n`` most-recently-added warmstart
        games. Returns the number evicted.
        """
        warm_idxs = [i for i, g in enumerate(self.buffer) if bool(g.external_values)]
        drop = set(warm_idxs[: max(0, len(warm_idxs) - n)])
        if not drop:
            return 0
        self.buffer = [g for i, g in enumerate(self.buffer) if i not in drop]
        self._priorities = [p for i, p in enumerate(self._priorities) if i not in drop]
        return len(drop)

    def sample_batch(
        self,
        batch_size: int,
        num_unroll_steps: int,
        td_steps: int,
        discount: float,
        action_space_size: int,
        alpha: float = 0.0,
        beta: float = 1.0,
        value_head_type: str = "support",
        history_frames: int = 1,
        eval_to_wdl_alpha: float = 4.0,
        eval_to_wdl_beta: float = 2.0,
        warmstart_sample_frac: float = 0.0,
        decisive_sample_frac: float = 0.0,
        batch_mixture: dict | None = None,
        position_sampling: str = "per_game",
        q_ratio: float = 0.0,
        warmstart_q_ratio: float | None = None,
        selfplay_q_ratio: float | None = None,
        repetition_penalty: float = 0.0,
        repetition_penalty_window: int = 0,
        repetition_penalty_decay: float = 0.0,
        material_value_weight: float = 0.0,
        material_value_scale: float = 5.0,
        tb_value_weight: float = 0.0,
        tb_moves_left_weight: float = 0.0,
        tb_policy_weight: float = 0.0,
        tb_value_hard: bool = False,
        build_legal_masks: bool = False,
        build_material_target: bool = False,
        symmetry_augment: bool = False,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Sample a batch of positions from stored games using PER.

        Returns:
            batch: dict of batched tensors
            game_indices: indices into self.buffer for priority updates
            is_weights: importance sampling weights to correct for PER bias
        """
        n = len(self.buffer)
        priorities = np.array(self._priorities[:n], dtype=np.float64)
        # Repair any non-finite priorities from past NaN/Inf TD errors.
        priorities = np.where(np.isfinite(priorities) & (priorities > 0), priorities, 1.0)
        probs = priorities ** alpha
        probs /= probs.sum()

        # Stratified sampling: oversample a "priority subset" of games into every
        # batch so a specific signal can't be flooded out by the majority class.
        #   Mode A (warmstart_sample_frac>0): subset = warmstart games
        #     (external_values populated); complement = self-play. Keeps a
        #     Stockfish anchor in every batch (catastrophic-forgetting guard).
        #   Mode B (decisive_sample_frac>0): subset = DECISIVE self-play games
        #     (|game_outcome|>=0.5, i.e. |z|=1); complement = the rest (draws +
        #     any warmstart). Keeps a NON-CONSTANT value signal (|z|=1 targets)
        #     in every batch against the draw-saturation loop, where a rising
        #     self-play draw rate makes the z=0 majority wash out the value
        #     gradient and the value head collapses to predicting "draw".
        # Modes are mutually exclusive (warmstart takes precedence). Both fall
        # back to flat PER when frac=0 or a stratum is empty.
        _mix_indices = None
        _mix_weights = None
        if batch_mixture:
            # CHANNEL MIXTURE (task #19, unified buffer): declarative batch
            # composition over GameHistory.channel strata. Supersedes the
            # two-stratum modes below (they still run for RNG-stream/back-compat
            # but their picks are overridden). Mixture weights renormalize over
            # channels that actually hold games (during warmup the selfplay
            # channel is empty -> its share redistributes proportionally, so
            # composition stays a CHOSEN ratio instead of an emergent one).
            # Per-stratum PER probs + IS weights mirror the legacy math.
            ch_indices: dict[str, list[int]] = {}
            for i in range(n):
                ch_indices.setdefault(self.buffer[i].channel, []).append(i)
            active = {c: np.asarray(v, dtype=np.int64)
                      for c, v in ch_indices.items()
                      if v and float(batch_mixture.get(c, 0.0)) > 0.0}
            if active:
                tot_w = sum(float(batch_mixture[c]) for c in active)
                # Largest-remainder rounding so counts hit batch_size exactly.
                raw = {c: batch_size * float(batch_mixture[c]) / tot_w for c in active}
                counts = {c: int(raw[c]) for c in active}
                rem = batch_size - sum(counts.values())
                for c in sorted(active, key=lambda c: raw[c] - int(raw[c]), reverse=True):
                    if rem <= 0:
                        break
                    counts[c] += 1
                    rem -= 1
                idx_parts, w_parts = [], []
                for c, idx_arr in active.items():
                    k = counts[c]
                    if k <= 0:
                        continue
                    p = probs[idx_arr]
                    if position_sampling == "per_ply":
                        # Weight games by length so every stored PLY is equally
                        # likely — removes the hidden overweighting of short
                        # (anchor) games under per-game-uniform sampling.
                        p = p * np.array([max(len(self.buffer[i]), 1) for i in idx_arr],
                                         dtype=np.float64)
                    p = p / p.sum()
                    picked = np.random.choice(len(idx_arr), size=k, p=p)
                    idx_parts.append(idx_arr[picked])
                    w_parts.append((len(idx_arr) * p[picked]) ** (-beta))
                _mix_indices = np.concatenate(idx_parts)
                _mix_weights = np.concatenate(w_parts)
                _mix_weights = (_mix_weights / _mix_weights.max()).astype(np.float32)

        subset_idx_arr = None
        comp_idx_arr = None
        strat_frac = 0.0
        if warmstart_sample_frac > 0.0:
            strat_frac = warmstart_sample_frac
            subset_idx_arr = np.array(
                [i for i in range(n) if self.buffer[i].external_values], dtype=np.int64)
            comp_idx_arr = np.array(
                [i for i in range(n) if not self.buffer[i].external_values], dtype=np.int64)
        elif decisive_sample_frac > 0.0:
            strat_frac = decisive_sample_frac
            subset_idx_arr = np.array(
                [i for i in range(n)
                 if abs(self.buffer[i].game_outcome) >= 0.5 and not self.buffer[i].external_values],
                dtype=np.int64)
            comp_idx_arr = np.array(
                [i for i in range(n)
                 if not (abs(self.buffer[i].game_outcome) >= 0.5 and not self.buffer[i].external_values)],
                dtype=np.int64)

        if (subset_idx_arr is not None and len(subset_idx_arr) > 0
                and len(comp_idx_arr) > 0):
            n_sub = int(round(batch_size * strat_frac))
            n_comp = batch_size - n_sub

            sub_probs = probs[subset_idx_arr]
            sub_probs = sub_probs / sub_probs.sum()
            comp_probs = probs[comp_idx_arr]
            comp_probs = comp_probs / comp_probs.sum()

            picked_sub_local = np.random.choice(len(subset_idx_arr), size=n_sub, p=sub_probs)
            picked_comp_local = np.random.choice(len(comp_idx_arr), size=n_comp, p=comp_probs)
            game_indices = np.concatenate([
                subset_idx_arr[picked_sub_local],
                comp_idx_arr[picked_comp_local],
            ])
            # IS weights against the per-stratum sampling probabilities.
            # weight_i = 1 / (k_stratum * P_stratum(i)), then normalize across batch.
            weights = np.empty(batch_size, dtype=np.float64)
            weights[:n_sub] = (len(subset_idx_arr) * sub_probs[picked_sub_local]) ** (-beta)
            weights[n_sub:] = (len(comp_idx_arr) * comp_probs[picked_comp_local]) ** (-beta)
            weights = (weights / weights.max()).astype(np.float32)
        else:
            game_indices = np.random.choice(n, size=batch_size, p=probs)
            # Importance sampling weights: w_i = (1 / (N * P(i)))^beta, normalised
            weights = (n * probs[game_indices]) ** (-beta)
            weights = (weights / weights.max()).astype(np.float32)

        if _mix_indices is not None:
            # Channel mixture supersedes the legacy pick (see block above).
            game_indices = _mix_indices
            weights = _mix_weights

        target_observations = []
        target_obs_masks = []
        actions_batch = []
        values_batch = []
        rewards_batch = []
        policies_batch = []
        legal_masks_batch = []
        moves_left_batch = []
        material_batch = []
        sym_reward_keep = []

        for g_idx in game_indices:
            game = self.buffer[g_idx]
            pos = np.random.randint(0, len(game))
            obs_list, obs_mask, actions, values, rewards, policies = game.make_target(
                pos, num_unroll_steps, td_steps, discount, action_space_size,
                value_head_type=value_head_type,
                history_frames=history_frames,
                eval_to_wdl_alpha=eval_to_wdl_alpha,
                eval_to_wdl_beta=eval_to_wdl_beta,
                q_ratio=q_ratio,
                warmstart_q_ratio=warmstart_q_ratio,
                selfplay_q_ratio=selfplay_q_ratio,
                repetition_penalty=repetition_penalty,
                repetition_penalty_window=repetition_penalty_window,
                repetition_penalty_decay=repetition_penalty_decay,
                material_value_weight=material_value_weight,
                material_value_scale=material_value_scale,
                tb_value_weight=tb_value_weight,
                tb_policy_weight=tb_policy_weight,
                tb_value_hard=tb_value_hard,
            )
            # D4 symmetry augmentation (src/training/symmetry.py): for pawnless,
            # castle-free windows, apply ONE random group element to the whole
            # sample — obs frames, unroll actions, policy targets (and the legal
            # masks below) — value/DTM/material targets are invariants. Forces
            # relative-geometry features over absolute-square memorization.
            sym_g = 0
            if symmetry_augment and game.game_name == "chess":
                from .symmetry import (SAFE_ELEMENTS, transform_actions,
                                       transform_obs, transform_policy_dense,
                                       window_eligible)
                if window_eligible(obs_list):
                    # Draw from the parity-safe subgroup ONLY — see
                    # symmetry.SAFE_ELEMENTS. R90/R270/diagonals corrupt
                    # cross-ply frame consistency under STM encoding.
                    sym_g = int(SAFE_ELEMENTS[np.random.randint(0, len(SAFE_ELEMENTS))])
                    if sym_g:
                        obs_list = [transform_obs(o, sym_g) for o in obs_list]
                        actions = transform_actions(actions, sym_g)
                        policies = [transform_policy_dense(p, sym_g) for p in policies]
            # REWARD-PRECISION GUARD (2026-07-05 root cause): the reward head is
            # the search's in-tree mate detector and THE conversion-critical
            # component (muted-reward diag: 0.20 -> 0.048 conversion on the same
            # weights). Training it on transformed views traded precision for
            # recall (false mate-fire 0.12 -> 0.41), poisoning backups in winning
            # subtrees. Reward learns from CANONICAL views only; the trainer
            # renormalizes kept samples to preserve gradient magnitude. All
            # other heads keep the full invariance pressure.
            sym_reward_keep.append(0.0 if sym_g else 1.0)
            target_observations.append(torch.stack(obs_list))  # (K+1, C, H, W)
            target_obs_masks.append(obs_mask)
            actions_batch.append(actions)
            values_batch.append(values)
            rewards_batch.append(rewards)
            policies_batch.append(policies)
            # Moves-left target: plies-remaining at each unroll step. Trivial fn of
            # position indices (no make_target ripple); POV-independent; past-end
            # steps clamp to 0 and are zeroed by target_obs_mask in the loss.
            final_idx = len(game) - 1
            tbml = getattr(game, "tablebase_moves_left", None)
            ml_row = []
            for i in range(num_unroll_steps + 1):
                idx = pos + i
                base = float(max(0, final_idx - idx))
                # DTM relabel (Lc0 Gaviota MLH rescoring): at in-TB decisive plies,
                # trust the tablebase plies-to-mate over the policy-rollout plies-to-
                # end (which is the shuffle length in unconverted endgames). NaN entries
                # (draws / out of TB) keep the rollout target. weight 1.0 = full replace.
                if (tb_moves_left_weight > 0.0 and tbml and idx < len(tbml)
                        and tbml[idx] == tbml[idx]):
                    base = (1.0 - tb_moves_left_weight) * base + tb_moves_left_weight * float(tbml[idx])
                ml_row.append(base)
            moves_left_batch.append(ml_row)
            if build_material_target:
                # Current STM material margin at each unrolled position (pos+i),
                # from that position's own observation (already STM-relative).
                # chess only; past-end / non-chess steps → 0.0 (zeroed by the
                # obs mask in the loss anyway).
                is_chess = (game.game_name == "chess")
                row = []
                for i in range(num_unroll_steps + 1):
                    idx = pos + i
                    if is_chess and idx < len(game.observations):
                        row.append(_material_margin_stm(game.observations[idx]))
                    else:
                        row.append(0.0)
                material_batch.append(row)
            if build_legal_masks:
                # (K+1, A) legal-move mask aligned with policies/actions: 1.0 on
                # legal moves at ply (pos + i), 0.0 on illegal. Mirrors make_target's
                # idx = pos + i. Where legality is unknown (past game end, or a ply
                # without a recorded legal set — e.g. some warmstart games), fall
                # back to all-ones so that ply behaves exactly like the unmasked
                # (standard) path rather than masking everything out.
                lm = np.ones((num_unroll_steps + 1, action_space_size), dtype=np.float32)
                for i in range(num_unroll_steps + 1):
                    idx = pos + i
                    if idx < len(game) and idx < len(game.legal_actions_list):
                        legal = game.legal_actions_list[idx]
                        if legal:
                            row = np.zeros(action_space_size, dtype=np.float32)
                            row[np.asarray(legal, dtype=np.int64)] = 1.0
                            lm[i] = row
                if sym_g:
                    from .symmetry import transform_legal_mask
                    for i in range(num_unroll_steps + 1):
                        lm[i] = transform_legal_mask(lm[i], sym_g)
                legal_masks_batch.append(lm)

        target_observations_t = torch.stack(target_observations)  # (B, K+1, C, H, W)

        # Per-sample warmstart membership — lets the trainer log per-stratum
        # losses (warmstart vs self-play) so we can detect lever-2 health
        # (warmstart loss → 0 while self-play diverges signals overfitting on
        # the anchor; both staying high signals the anchor is too small).
        is_warmstart = np.array(
            [bool(self.buffer[i].external_values) for i in game_indices],
            dtype=bool,
        )
        # Per-sample channel ids for composition logging + per-channel loss
        # splits (task #19): 0=warmstart, 1=anchor, 2=selfplay.
        _CH_ID = {"warmstart": 0, "anchor": 1, "selfplay": 2}
        channel_ids = np.array(
            [_CH_ID[self.buffer[i].channel] for i in game_indices], dtype=np.int64)

        batch = {
            # Root obs (k=0) kept at "observations" for back-compat with callers that
            # only need the initial inference input.
            "observations": target_observations_t[:, 0],
            "target_observations": target_observations_t,
            "target_obs_mask": torch.tensor(target_obs_masks, dtype=torch.float32),
            "actions": torch.tensor(actions_batch, dtype=torch.long),
            # values_batch may be list-of-list-of-floats (support head) or
            # list-of-list-of-(3,)-arrays (WDL head). np.asarray handles both
            # shapes and avoids the slow per-element conversion warning.
            "target_values": torch.from_numpy(np.asarray(values_batch, dtype=np.float32)),
            "target_rewards": torch.tensor(rewards_batch, dtype=torch.float32),
            "target_policies": torch.from_numpy(np.stack(policies_batch)),
            "target_moves_left": torch.tensor(moves_left_batch, dtype=torch.float32),  # (B, K+1) plies-to-end
            "is_warmstart": torch.from_numpy(is_warmstart),
            "channel_ids": torch.from_numpy(channel_ids),
        }
        if symmetry_augment:
            # 1.0 = canonical view (reward loss active), 0.0 = transformed
            # (reward loss masked; see reward-precision guard above).
            batch["sym_reward_keep"] = torch.tensor(sym_reward_keep, dtype=torch.float32)
        if build_material_target:
            batch["target_material"] = torch.tensor(material_batch, dtype=torch.float32)  # (B, K+1) STM material margin
        if build_legal_masks:
            # (B, K+1, A) — 1.0 legal, 0.0 illegal. Consumed by the trainer's
            # masked policy loss + illegal-mass penalty.
            batch["target_legal_masks"] = torch.from_numpy(np.stack(legal_masks_batch))
        return batch, game_indices, weights

    def update_priorities(self, game_indices: np.ndarray, td_errors: np.ndarray, epsilon: float = 1e-6):
        """Update game priorities from TD errors computed during training."""
        for idx, err in zip(game_indices, td_errors):
            if 0 <= idx < len(self._priorities):
                err = float(abs(err))
                # Guard against NaN/Inf from divergent training steps poisoning the buffer.
                if not np.isfinite(err):
                    err = 1.0
                self._priorities[idx] = err + epsilon

    def save(self, path: str | Path, filter_fn=None, max_games: int | None = None,
             format_version: int = 3):
        """Stream-pickle buffer to path, one record per game.

        Writes a header dict first, then one record pickle per game. Peak
        memory is ~one game instead of the whole buffer, which matters when
        the buffer is multi-GB: the old single-dump path allocated the
        entire serialized payload in RAM before writing and OOM'd the host.

        format_version:
            2 — legacy streaming ``(GameHistory, priority)`` pairs.
            3 — compact streaming ``(compact_dict, priority)`` pairs (default).
                Drops observations + legal_actions_list (replayable on load),
                sparsifies policies. Typically ~100× smaller than v2.
                Requires ``game`` when loading.

        filter_fn: optional ``(GameHistory) -> bool``. Only matching games
        are written; priorities stay aligned.
        max_games: if set, keep only the most recent ``max_games`` (post-filter)
        records. total_games is preserved regardless so the training-progress
        counter survives capped saves.
        """
        if format_version not in (2, 3):
            raise ValueError(f"Unsupported format_version {format_version}; expected 2 or 3")
        if filter_fn is None:
            pairs = list(zip(self.buffer, self._priorities))
        else:
            pairs = [(g, p) for g, p in zip(self.buffer, self._priorities) if filter_fn(g)]
        if max_games is not None and len(pairs) > max_games:
            pairs = pairs[-max_games:]
        header = {
            "version": format_version,
            "total_games": self.total_games,
            "n_records": len(pairs),
        }
        with open(path, "wb") as f:
            pickle.dump(header, f)
            if format_version == 3:
                for game, priority in pairs:
                    pickle.dump((game.to_compact_dict(), priority), f)
            else:
                for game, priority in pairs:
                    pickle.dump((game, priority), f)

    def load(self, path: str | Path, game=None):
        """Load buffer from path. Dispatches on header version.

        Supported:
          * v2 — streaming header + ``(GameHistory, priority)`` pairs.
          * v3 — streaming header + ``(compact_dict, priority)`` pairs;
            requires ``game`` so ``GameHistory.from_compact_dict`` can replay
            actions to rebuild observations.
        """
        with open(path, "rb") as f:
            first = pickle.load(f)
            if not (isinstance(first, dict) and first.get("version") in (2, 3)):
                raise ValueError(
                    f"{path}: unrecognized buffer format "
                    f"(expected v2 or v3 streaming header, got {type(first).__name__})"
                )
            version = first["version"]
            if version == 3 and game is None:
                raise ValueError(
                    f"{path}: v3 (compact) buffer requires a `game` argument "
                    f"to reconstruct observations."
                )
            self.buffer = []
            self._priorities = []
            self.total_games = first["total_games"]
            for _ in range(first["n_records"]):
                record, priority = pickle.load(f)
                if version == 3:
                    record = GameHistory.from_compact_dict(record, game)
                self.buffer.append(record)
                self._priorities.append(priority)
            # Reassign monotonic birth stamps in load order (relative age preserved)
            # so retention-weighted eviction works after a resume.
            for i, g in enumerate(self.buffer):
                g.birth = i
            self._save_counter = len(self.buffer)

    def load_warmstart_games(self, paths: list[Path | str], game=None) -> int:
        """Load pickled shards of GameHistory objects into the buffer.

        Supported shard layouts:
          * Legacy — pickle of ``list[GameHistory]``.
          * Compact (v2) — header dict ``{"version": 2, "n_records": N}``
            followed by N compact-dict pickles. Requires ``game`` for
            reconstruction.

        Games are inserted via save_game so priorities are seeded to the
        current max. Returns the number of games loaded.
        """
        loaded = 0
        for p in paths:
            for g in _iter_shard_games(p, game):
                self.save_game(g)
                loaded += 1
        return loaded

    def sample_games_for_reanalyze(self, n: int) -> tuple[np.ndarray, list]:
        """Sample n games uniformly (ignoring PER priorities) for reanalyze.

        Uniform sampling avoids double-biasing: PER already up-weights high-error
        positions during training; using it here too would over-concentrate reanalyze
        on the same games. See muzero-general replay_buffer.py (force_uniform=True).

        Warmstart games (``external_values`` populated) are excluded: their policy
        targets are supervised one-hots from Stockfish, and their value targets come
        from external_values directly — reanalyze would strictly degrade both.
        TB-authored games (anchor / rollout-filled — ``tb_authored``) are excluded
        for the same reason: their policy targets are tablebase demonstrations;
        reanalyze would overwrite perfect targets with the net's own weaker visits.
        """
        eligible = [
            i for i, g in enumerate(self.buffer)
            if not g.external_values and not getattr(g, "tb_authored", False)
        ]
        if not eligible:
            return np.array([], dtype=np.int64), []
        n = min(n, len(eligible))
        indices = np.random.choice(eligible, size=n, replace=False)
        return indices, [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)

    def nbytes(self) -> dict[str, int]:
        """Return per-component byte counts for the live buffer.

        Helps diagnose whether RSS growth tracks logical buffer size or
        something else (view aliasing, fragmentation, retained accumulators).

        Returns dict with keys ``observations``, ``policies``, ``legal_actions``,
        ``scalars`` (actions/rewards/values/etc.), ``total``, plus a
        ``view_parents`` field that sums the size of unique numpy ``.base``
        arrays referenced by ``policies`` slices — if non-zero, slices are
        pinning their parent buffers (the self_play.py:585/587 leak signature).
        """
        obs_b = pol_b = legal_b = scalar_b = 0
        seen_parents: dict[int, int] = {}
        for g in self.buffer:
            for o in g.observations:
                obs_b += o.element_size() * o.numel()
            for p in g.policies:
                if p is None:
                    continue
                if isinstance(p, tuple):  # sparse (idx, val) — no parent pinning
                    idx, val = p
                    pol_b += idx.nbytes + val.nbytes
                elif p.size:
                    pol_b += p.nbytes
                    base = p.base
                    if base is not None:
                        seen_parents.setdefault(id(base), base.nbytes)
            for l in g.legal_actions_list:
                legal_b += 28 * len(l)  # python int overhead
            scalar_b += 8 * (len(g.actions) + len(g.rewards) + len(g.root_values))
        view_parents_b = sum(seen_parents.values())
        return {
            "observations": obs_b,
            "policies": pol_b,
            "legal_actions": legal_b,
            "scalars": scalar_b,
            "total": obs_b + pol_b + legal_b + scalar_b,
            "view_parents": view_parents_b,
        }
