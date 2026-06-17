"""Tensor-native MCTS for MuZero — GPU-resident search tree.

Companion to ``src/mcts/mcts.py`` (the existing Python+numpy ``BatchedMCTS``).
Same outer interface — ``run_batch(observations, legal_actions_list,
add_noise=True)`` returns a list of ``MCTSNode``-equivalent objects so callers
in ``src/training/self_play.py`` can swap implementations with no further
changes.

Internally everything per-simulation runs on GPU tensors:

- 0 GPU↔CPU syncs per simulation (vs. 3 in ``BatchedMCTS``).
- 1 sync per ply (compat shim copies root data back when run_batch returns).

v1 scope (per ``plan_tensor_mcts_implementation.md``):

- Sampled MuZero §5.1 Proposed Modification at root and leaves.
- Mover-POV value/reward sign convention (matches ``BatchedMCTS``):
    Q(parent) = reward + γ·V(child)_parent_POV = reward − γ·child.value_child_POV
- Root Dirichlet noise applied after the K-action sample.
- MinMaxStats Q-normalization per game.
- v1 dedup choice: keep duplicates with β̂(slot) = 1/K (option (b) from the
  plan). Mathematically equivalent to count(a)/K in the limit; PUCT may visit
  the same action across multiple slots, materializing parallel copies of the
  same conceptual subtree. Dedup happens in the compat shim before priors and
  visit counts are exposed to ``select_action``.

Out of scope for v1:

- Gumbel root (Sequential Halving). Caller must keep ``use_gumbel=False``.
- Full-action root expansion. ``config.sample_k`` must be set; for tiny action
  spaces fall back to the existing ``BatchedMCTS``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from src.mcts.mcts import MCTSNode


_NEG_INF = float("-inf")


def select_action_gpu(
    child_actions: torch.Tensor,
    child_visits: torch.Tensor,
    temperature: torch.Tensor,
    action_space_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-resident action selection from MCTS visit counts.

    Replaces the pattern ``select_action(MCTSNode, temperature)`` for the
    GPU-resident self-play path. Stays on the GPU throughout — no per-game
    Python loops, no .cpu() transfers.

    Args:
        child_actions: [N, K] int32 — sampled actions at root (may contain
            -1 padding and option-(b) duplicates).
        child_visits: [N, K] int32 — visit count per slot.
        temperature: [N] float32 — per-game temperature (0 for greedy,
            >0 for stochastic). Pass a 0-d tensor to broadcast.
        action_space_size: A — width of the dense policy output.

    Returns:
        action: [N] int64 — chosen action per game.
        dense_policy: [N, A] float32 — RAW deduped visit fraction
            N(a)/ΣN(a) (option-(b) duplicates summed per unique action via
            scatter_add). Temperature-independent training target for ALL
            temperatures, including greedy (T=0); see bug_hunt_2026_06_13.md
            §A+§B. Temperature drives only ``action`` selection.
    """
    n, K = child_actions.shape
    dev = child_actions.device

    if temperature.dim() == 0:
        temperature = temperature.expand(n)

    visits_f = child_visits.to(torch.float32)
    valid = child_actions != -1                                       # [N, K]
    visits_f = torch.where(valid, visits_f, torch.zeros_like(visits_f))

    is_greedy = temperature == 0                                       # [N]

    # bug_hunt_2026_06_13.md §A: SUM the per-slot visits into a per-unique-action
    # dense tensor FIRST (option-(b) duplicates split an action's visits across
    # many of the K slots), then apply temperature / build the target on the
    # deduped action-level visits — matching the numpy _build_compat_roots path.
    # Doing temperature per-slot (softmax over raw K slots) corrupts both the
    # training target and the move choice for T≠1 since softmax is nonlinear.
    actions_long = child_actions.clamp(min=0).long()                   # [N, K]
    masked_visits = torch.where(valid, visits_f, torch.zeros_like(visits_f))
    dense_visits = torch.zeros(n, action_space_size, device=dev, dtype=torch.float32)
    dense_visits.scatter_add_(1, actions_long, masked_visits)          # [N, A]

    # Dense TRAINING TARGET: raw deduped visit fraction N(a)/ΣN(a), independent
    # of temperature (bug_hunt §B). Rows with no visits fall back to all-zeros.
    visit_total = dense_visits.sum(dim=1, keepdim=True)                # [N, 1]
    dense_target = dense_visits / visit_total.clamp(min=1e-9)          # [N, A]

    # Greedy path: argmax over summed action visits.
    greedy_action = dense_visits.argmax(dim=1)                         # [N]

    # Sampled path: softmax(log(summed_visits) / T) over UNIQUE actions.
    # log-space is numerically stable for high inv_temp; zero-visit actions get
    # -inf logits → 0 prob.
    action_valid = dense_visits > 0                                    # [N, A]
    safe_temp = temperature.clamp(min=1e-3)
    inv_temp = (1.0 / safe_temp).unsqueeze(1)                          # [N, 1]
    log_visits = torch.log(dense_visits.clamp(min=1e-9))               # [N, A]
    action_logits = log_visits * inv_temp                              # [N, A]
    action_logits = torch.where(
        action_valid, action_logits, torch.full_like(action_logits, _NEG_INF)
    )
    sample_probs = torch.softmax(action_logits, dim=1)                 # [N, A]
    # Safety net: a row with no valid action (all-invalid root) or NaN softmax
    # falls back to uniform over valid actions (or all if none).
    has_any = action_valid.any(dim=1, keepdim=True)
    safe_probs = torch.where(
        action_valid,
        torch.ones_like(sample_probs),
        torch.zeros_like(sample_probs),
    )
    safe_probs = safe_probs / safe_probs.sum(dim=1, keepdim=True).clamp(min=1.0)
    nan_mask = torch.isnan(sample_probs).any(dim=1, keepdim=True) | ~has_any
    sample_probs = torch.where(nan_mask, safe_probs, sample_probs)

    sampled_action = torch.multinomial(sample_probs, 1).squeeze(1)     # [N]

    action = torch.where(is_greedy, greedy_action, sampled_action).long()

    return action, dense_target


class TensorMCTS:
    """Sampled MuZero tensor-native MCTS.

    Tree storage is pre-allocated on first ``run_batch`` and reused across
    plies; re-allocates if N, M, or hidden shape changes.

    Args:
        network: MuZero network with ``initial_inference`` /
            ``recurrent_inference`` returning scalar value/reward.
        game: provides ``action_space_size``.
        config: ``MuZeroConfig``-like (reads ``num_simulations``, ``sample_k``,
            ``discount``, ``dirichlet_alpha``, ``dirichlet_epsilon``).
        device: torch device for the search tree (typically same as network).
        hidden_dtype: storage dtype for ``node_hidden``. Default fp32; cast to
            fp16 when memory pressure matters (chess preset @ N=256/M=401 needs
            ~6.7 GB fp32 for the hidden tensor alone, ~3.4 GB at fp16).
    """

    PB_C_BASE = 19652
    PB_C_INIT = 1.25
    # Hard cap on selection depth. The PUCT walk in ``_select`` runs at most
    # this many iterations per simulation, regardless of ``num_simulations``.
    # Why: in a Sampled MuZero tree (K=50, sims=200) average depth is ~5-10
    # and max depth rarely exceeds ~16 (log_K scaling). Capping at 32 halves
    # the unrolled compile work vs M-1=200 with negligible correctness risk.
    # Raise this if you observe truncated paths (the safety branch at the
    # end of ``_select`` force-stops any still-walking game at the cap).
    MAX_SELECT_DEPTH = 32

    def __init__(
        self,
        network,
        game,
        config,
        device: str | torch.device = "cuda",
        hidden_dtype: torch.dtype = torch.float32,
        compile: bool = True,
        amp_dtype: torch.dtype | None = None,
        select_backend: str = "compile",
        use_subtree_reuse: bool = False,
    ):
        self.network = network
        self.game = game
        self.config = config
        self.device = torch.device(device)
        self.hidden_dtype = hidden_dtype
        # Inference autocast for the network calls inside MCTS (initial +
        # recurrent inference). Tree storage stays at hidden_dtype; only the
        # network forward gets the autocast benefit. Set ``amp_dtype=None`` to
        # disable. fp16 is the standard choice for inference; values/rewards
        # in [-1, 1] keep enough precision.
        self.amp_dtype = amp_dtype if self.device.type == "cuda" else None
        # Inference-time global settings. cudnn.benchmark auto-tunes conv
        # algorithms for the network's fixed shapes; TF32 enables Ampere's
        # fast matmul path for fp32 ops. Both are no-ops for non-CUDA.
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # Whether to wrap _select / _backprop with torch.compile. Each iteration
        # of the per-step selection loop fires ~25 small kernels (gathers, PUCT,
        # argmax, writes); torch.compile fuses them via reduce-overhead mode +
        # CUDA graphs, cutting per-iteration kernel-launch overhead by 5-10×.
        # Set False when debugging to read clean tracebacks.
        self.compile = compile and self.device.type == "cuda"
        # Selection backend: "compile" (default, torch.compile + inductor),
        # "triton" (custom fused PUCT-step kernel — see tensor_mcts_triton.py),
        # or "eager" (plain PyTorch, for debugging).
        if select_backend not in ("compile", "triton", "eager"):
            raise ValueError(
                f"Unknown select_backend={select_backend!r}; "
                f"expected 'compile', 'triton', or 'eager'."
            )
        if select_backend == "triton" and self.device.type != "cuda":
            raise ValueError("select_backend='triton' requires CUDA.")
        self.select_backend = select_backend

        self.action_space_size = int(game.action_space_size)
        sample_k = getattr(config, "sample_k", None)
        if sample_k is None or int(sample_k) >= self.action_space_size:
            # Fall through to "K = action_space"; the multinomial path still
            # works (sampling from a categorical of size A with K=A and
            # replacement). Callers with tiny action spaces should usually
            # prefer BatchedMCTS, which expands all legal actions at root.
            sample_k = self.action_space_size
        self.K = int(sample_k)

        # Tree storage — populated by _allocate on first run_batch.
        self._allocated_n = -1
        self._allocated_m = -1
        self._allocated_hidden_shape: tuple[int, int, int] | None = None
        self._cached_compute_dtype: torch.dtype | None = None
        # Cached compiled callables (set after first _allocate so dynamo sees
        # static shapes in tracing).
        self._select_step_compiled = None
        self._backprop_compiled = None
        # Subtree reuse: when True, the tree storage is sized to fit the
        # carried-over subtree PLUS one ply of fresh sims. Without this, the
        # post-advance + new-sims node count overflows M = num_sims + 1.
        self.use_subtree_reuse = bool(use_subtree_reuse)
        # ``has_prior_search``: when True, ``run_batch_gpu`` skips
        # ``_initialize_root`` and assumes slot 0 already holds the new root
        # (set by a prior ``advance_root``). Reset by ``reset_search`` or by
        # ``advance_root`` itself when reuse fails.
        self.has_prior_search = False

    def _compute_dtype(self) -> torch.dtype:
        """Network's parameter dtype — used to cast inputs before recurrent_inference.

        Falls back to ``hidden_dtype`` when the stub network has no parameters
        (test fixtures). Cached on first call.
        """
        if self._cached_compute_dtype is not None:
            return self._cached_compute_dtype
        first_param = next(self.network.parameters(), None)
        self._cached_compute_dtype = (
            first_param.dtype if first_param is not None else self.hidden_dtype
        )
        return self._cached_compute_dtype

    def _amp_ctx(self):
        """Autocast context for the network forward calls. No-op when disabled
        or on CPU. Cast happens at the network's compute boundary; tree
        storage stays at hidden_dtype."""
        if self.amp_dtype is None:
            return torch.amp.autocast(
                device_type=self.device.type, enabled=False
            )
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=True,
        )

    # ------------------------------------------------------------------ alloc

    def _allocate(self, n: int, num_simulations: int, hidden_shape: tuple[int, int, int]):
        """Allocate (or reuse) tree tensors.

        M = num_simulations + 1 (root + sims) without subtree reuse;
        M = 2*num_simulations + 1 when ``use_subtree_reuse`` is on, so the
        carried-over subtree (≤ num_simulations nodes from prior search)
        plus a fresh ply's num_simulations new nodes both fit. If the
        carry-over plus new sims would still exceed M, ``advance_root``
        falls back to a fresh search for safety.
        """
        m = (2 * num_simulations + 1) if self.use_subtree_reuse else (num_simulations + 1)
        if (
            n == self._allocated_n
            and m == self._allocated_m
            and hidden_shape == self._allocated_hidden_shape
        ):
            return

        c, h, w = hidden_shape
        dev = self.device
        K = self.K

        # Per-node tensors.
        self.node_count = torch.zeros(n, dtype=torch.int32, device=dev)
        self.node_visits = torch.zeros(n, m, dtype=torch.int32, device=dev)
        self.node_value_sum = torch.zeros(n, m, dtype=torch.float32, device=dev)
        self.node_reward = torch.zeros(n, m, dtype=torch.float32, device=dev)
        self.node_hidden = torch.zeros(n, m, c, h, w, dtype=self.hidden_dtype, device=dev)
        self.parent_idx = torch.full((n, m), -1, dtype=torch.int32, device=dev)
        self.parent_child_slot = torch.full((n, m), -1, dtype=torch.int32, device=dev)

        # Per-(node, slot) tensors. Mirror node-level stats so PUCT can read
        # them with one [N, K] gather instead of double-indirection through
        # child_node_idx (which would pull stats from a -1 sentinel for
        # unmaterialized slots anyway).
        self.child_actions = torch.full((n, m, K), -1, dtype=torch.int32, device=dev)
        self.child_priors = torch.zeros(n, m, K, dtype=torch.float32, device=dev)
        self.child_visits = torch.zeros(n, m, K, dtype=torch.int32, device=dev)
        self.child_value_sum = torch.zeros(n, m, K, dtype=torch.float32, device=dev)
        self.child_rewards = torch.zeros(n, m, K, dtype=torch.float32, device=dev)
        self.child_node_idx = torch.full((n, m, K), -1, dtype=torch.int32, device=dev)

        # MinMaxStats per game. Sentinels: empty range = (+inf, -inf).
        self.mm_min = torch.full((n,), float("inf"), dtype=torch.float32, device=dev)
        self.mm_max = torch.full((n,), _NEG_INF, dtype=torch.float32, device=dev)

        # Reusable arange for advanced indexing.
        self._arange_n = torch.arange(n, device=dev)

        self._allocated_n = n
        self._allocated_m = m
        self._allocated_hidden_shape = hidden_shape

        # Compile per-sim methods. Backprop always uses torch.compile; the
        # selection path is dispatched via ``select_backend`` (compile|triton|eager).
        # ``mode='default'`` (inductor fusion, no cudagraphs) — see commit
        # b90640e for why cudagraphs is incompatible with our tree storage.
        if self.compile:
            self._backprop = torch.compile(  # type: ignore[method-assign]
                self.__class__._backprop.__get__(self),
                mode="default",
                dynamic=False,
                fullgraph=False,
            )
        if self.select_backend == "compile" and self.compile:
            # ``_select`` is monolithic — its full MAX_SELECT_DEPTH for-loop
            # body is unrolled into ONE inductor trace, which fuses the
            # 25+ ops/iter into a few kernels.
            self._select = torch.compile(  # type: ignore[method-assign]
                self.__class__._select.__get__(self),
                mode="default",
                dynamic=False,
                fullgraph=False,
            )
        elif self.select_backend == "triton":
            # Replace ``_select`` with the Triton-kernel-driven version.
            self._select = self._select_triton  # type: ignore[method-assign]

    def _reset(self):
        """Zero out tree tensors at the start of each MCTS run.

        Lazy reset (only touching slots up to last ply's max(node_count)) is
        a clear perf win at chess scale and a v1 follow-up; for now we just
        zero the whole tree on each call. Per-ply MCTS overhead is dominated
        by the network forward, not the reset.
        """
        self.node_count.zero_()
        self.node_visits.zero_()
        self.node_value_sum.zero_()
        self.node_reward.zero_()
        # node_hidden is overwritten on allocation; no need to zero.
        self.parent_idx.fill_(-1)
        self.parent_child_slot.fill_(-1)
        self.child_actions.fill_(-1)
        self.child_priors.zero_()
        self.child_visits.zero_()
        self.child_value_sum.zero_()
        self.child_rewards.zero_()
        self.child_node_idx.fill_(-1)
        self.mm_min.fill_(float("inf"))
        self.mm_max.fill_(_NEG_INF)

    # ---------------------------------------------------------------- public

    @torch.no_grad()
    def run_batch(
        self,
        observations: list[torch.Tensor],
        legal_actions_list: list[list[int]],
        add_noise: bool = True,
    ) -> list[MCTSNode]:
        """Run num_simulations MCTS sims for N games in parallel on GPU.

        Returns N ``MCTSNode`` objects (compat shim) populated with deduped
        ``child_actions`` / ``child_visits`` / ``child_priors`` so the
        existing ``select_action`` helpers work unchanged.
        """
        n = len(observations)
        num_sims = int(self.config.num_simulations)

        obs_batch = torch.stack(observations).to(self.device)
        with self._amp_ctx():
            hidden_batch, policy_logits_root, value_root = self.network.initial_inference(obs_batch)
        # hidden_batch: [N, C, H, W]; policy_logits_root: [N, A]; value_root: [N, 1] or [N].

        c, h, w = hidden_batch.shape[1:]
        self._allocate(n, num_sims, (c, h, w))
        self._reset()

        self._initialize_root(
            hidden_batch,
            policy_logits_root,
            legal_actions_list,
            add_noise,
        )

        for _ in range(num_sims):
            (
                path_node_idx,
                path_slot,
                leaf_parent_node,
                leaf_slot,
                leaf_action,
                depth,
            ) = self._select()

            new_node_idx, leaf_value = self._expand(
                leaf_parent_node,
                leaf_slot,
                leaf_action,
            )

            self._backprop(
                path_node_idx,
                path_slot,
                depth,
                new_node_idx,
                leaf_slot,
                leaf_value,
            )

        return self._build_compat_roots(n)

    @torch.no_grad()
    def run_batch_gpu(
        self,
        obs_batch: torch.Tensor,
        legal_mask: torch.Tensor,
        add_noise: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Sync-free variant of ``run_batch``.

        Args:
            obs_batch: [N, C, H, W] observation tensor on ``self.device``.
            legal_mask: [N, action_space] bool mask on ``self.device``.

        Returns a dict of GPU-resident tensors:

            ``child_actions``  : [N, K] int32 — sampled actions at root
                                 (may contain duplicates under option (b)).
            ``child_visits``   : [N, K] int32 — visit count per slot.
            ``child_priors``   : [N, K] float32 — β̂(slot) = 1/K + Dirichlet.
            ``root_value``     : [N] float32 — root mean Q (= value_sum/visits).

        Caller is responsible for any downstream aggregation (deduping
        actions, building dense policy, etc.) — kept GPU-side via
        :func:`select_action_gpu`.
        """
        # Compare types only — `cuda` and `cuda:0` are the same logical device
        # but compare unequal as torch.device objects.
        assert obs_batch.device.type == self.device.type, (
            f"obs_batch on {obs_batch.device}, expected {self.device}"
        )
        assert legal_mask.device.type == self.device.type, (
            f"legal_mask on {legal_mask.device}, expected {self.device}"
        )

        n = obs_batch.shape[0]
        num_sims = int(self.config.num_simulations)

        # Decide reuse viability up front. The carried-over root has K sampled
        # actions from the prior search's network-policy sample; some may be
        # illegal at the new state. If ANY game has zero legal sampled
        # actions, fall back to a fresh search (otherwise multinomial in
        # select_action_gpu hits all-zero probs).
        can_reuse = self.has_prior_search
        if can_reuse:
            root_actions = self.child_actions[:, 0, :]
            valid_slot = root_actions != -1
            action_long = root_actions.clamp(min=0).long()
            slot_legal = torch.gather(legal_mask, 1, action_long) & valid_slot
            any_legal = slot_legal.any(dim=1)
            if not bool(any_legal.all()):
                can_reuse = False

        if can_reuse:
            # Tree state holds the new root at slot 0 (compacted by prior
            # ``advance_root``). Skip initial_inference; refresh for the new ply.
            self._refresh_reused_root(legal_mask, add_noise)
        else:
            with self._amp_ctx():
                hidden_batch, policy_logits_root, _value_root = self.network.initial_inference(obs_batch)
            c, h, w = hidden_batch.shape[1:]
            self._allocate(n, num_sims, (c, h, w))
            self._reset()
            self._initialize_root_from_mask(
                hidden_batch, policy_logits_root, legal_mask, add_noise
            )
            self.has_prior_search = False

        for _ in range(num_sims):
            (
                path_node_idx,
                path_slot,
                leaf_parent_node,
                leaf_slot,
                leaf_action,
                depth,
            ) = self._select()
            new_node_idx, leaf_value = self._expand(
                leaf_parent_node, leaf_slot, leaf_action
            )
            self._backprop(
                path_node_idx,
                path_slot,
                depth,
                new_node_idx,
                leaf_slot,
                leaf_value,
            )

        root_visits_int = self.node_visits[:, 0]                # [N] int32
        root_visits_f = root_visits_int.to(torch.float32)
        root_value = self.node_value_sum[:, 0] / torch.clamp(
            root_visits_f, min=1.0
        )                                                        # [N] float32
        # Clone so the caller can hold these past the next run_batch call,
        # which would otherwise overwrite the underlying tree storage.
        return {
            "child_actions": self.child_actions[:, 0, :].clone(),  # [N, K] int32
            "child_visits": self.child_visits[:, 0, :].clone(),    # [N, K] int32
            "child_priors": self.child_priors[:, 0, :].clone(),    # [N, K] float32
            "root_value": root_value.clone(),                       # [N] float32
        }

    # ---------------------------------------------------------------- helpers

    def _initialize_root(
        self,
        hidden_batch: torch.Tensor,
        policy_logits_root: torch.Tensor,
        legal_actions_list: list[list[int]],
        add_noise: bool,
    ):
        """Build root node (slot 0); legal_actions arrives as CPU lists."""
        n = self._allocated_n
        action_space = self.action_space_size

        # Build legal mask CPU-side (one small N-loop + one transfer is fine
        # for the legacy compat path; the GPU-resident path bypasses this).
        legal_mask_cpu = torch.zeros(n, action_space, dtype=torch.bool)
        for g, legals in enumerate(legal_actions_list):
            if legals:
                legal_mask_cpu[g, legals] = True
        legal_mask = legal_mask_cpu.to(self.device)
        self._initialize_root_from_mask(
            hidden_batch, policy_logits_root, legal_mask, add_noise
        )

    @staticmethod
    def _dedup_sampled(sampled_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dedup [N, K] i.i.d.-sampled actions to UNIQUE actions per row.

        Sampled MuZero (Hubert 2021 §5.1) Proposed Modification: with β = π the
        PUCT prior over the unique sampled actions reduces to β̂(a) = count(a)/K.
        This mirrors the numpy oracle (`mcts.py::_sample_iid_with_replacement`,
        which uses `np.unique(..., return_counts=True)` and `count/K`), but keeps
        the fixed [N, K] slot width: each unique action occupies exactly ONE slot
        with prior = count(a)/K; the remaining (K − U) slots are padded with
        action = -1 / prior = 0 (PUCT masks action == -1 to -inf, so padding is
        never selected).

        Vectorized & fully on-device — no per-row Python loop, no CPU sync:
        sort each row, mark each value's first occurrence as its representative
        slot, count run-lengths via a segment-sum, and scatter the unique actions
        + counts back onto the representative slots.

        Args:
            sampled_actions: [N, K] int64, K i.i.d. samples (with replacement).

        Returns:
            actions: [N, K] int32 — unique actions on representative slots,
                -1 on padding slots.
            priors: [N, K] float32 — β̂(a) = count(a)/K on representative slots,
                0.0 on padding slots. Each row's non-pad priors sum to 1.
        """
        n, K = sampled_actions.shape
        dev = sampled_actions.device

        # Sort each row so equal actions are contiguous.
        sorted_actions, _ = torch.sort(sampled_actions, dim=1)            # [N, K]
        # A slot is the representative (first occurrence) of its value iff it
        # differs from its left neighbour. Slot 0 is always a representative.
        is_first = torch.ones(n, K, dtype=torch.bool, device=dev)
        is_first[:, 1:] = sorted_actions[:, 1:] != sorted_actions[:, :-1]

        # Run-length count per unique value, placed on its representative slot.
        # Each value spans [rep, next_rep); count = next_rep_index − rep_index.
        # rank[j] = number of representatives at or before slot j (1-indexed
        # group id). Representatives appear at the group-start positions; the
        # count of group g is (start of g+1) − (start of g). Compute via the
        # difference of representative slot indices.
        col = torch.arange(K, device=dev).unsqueeze(0).expand(n, K)       # [N, K]
        # Slot index of each representative; non-reps get K (sorted to the end).
        rep_pos = torch.where(is_first, col, torch.full_like(col, K))     # [N, K]
        rep_sorted, _ = torch.sort(rep_pos, dim=1)                        # [N, K]
        # next-start per representative (shift left); last real group ends at K.
        next_pos = torch.full_like(rep_sorted, K)
        next_pos[:, :-1] = rep_sorted[:, 1:]
        counts_packed = (next_pos - rep_sorted).clamp(min=0)              # [N, K]
        # rep_sorted lists representative slot indices (padding entries == K);
        # scatter the packed counts back onto those slots.
        valid_rep = rep_sorted < K
        rep_idx = rep_sorted.clamp(max=K - 1)                            # [N, K]
        counts = torch.zeros(n, K, dtype=torch.float32, device=dev)
        # scatter_ADD (not scatter_): padding reps (value K, clamped to K-1)
        # contribute 0 and must not clobber the real rep that legitimately
        # owns slot K-1.
        counts.scatter_add_(
            1, rep_idx,
            torch.where(valid_rep, counts_packed.to(torch.float32),
                        torch.zeros_like(counts_packed, dtype=torch.float32)),
        )

        actions = torch.where(
            is_first, sorted_actions,
            torch.full_like(sorted_actions, -1),
        ).to(torch.int32)                                                # [N, K]
        priors = counts / float(K)                                       # [N, K]
        return actions, priors

    def _enumerate_legal(
        self, legal_mask: torch.Tensor, root_probs: torch.Tensor
    ):
        """Pack each game's legal actions into the K root slots with their EXACT
        softmax prior — the no-sampling expansion used when a game has <= K legal
        actions. Returns ``(actions [N, K] int32, priors [N, K] float32)`` with
        unused slots = -1 / 0. Games with more than K legal actions get only
        their first K legal indices here (those games use the sampled path
        instead, so this truncated result is never read for them)."""
        n, A = legal_mask.shape
        K = self.K
        dev = self.device
        # rank[g, a] = 0-indexed slot for the a-th legal action in row g
        # (cumsum of the legal mask, minus 1). Illegal entries also get a rank
        # but are dropped by ``valid`` below.
        rank = torch.cumsum(legal_mask.to(torch.int64), dim=1) - 1       # [N, A]
        a_idx = torch.arange(A, device=dev).unsqueeze(0).expand(n, A)    # [N, A]
        valid = legal_mask & (rank >= 0) & (rank < K)                    # [N, A]
        # Scatter each legal action index into its rank slot. Invalid entries
        # are routed to a throwaway column K (sliced off after) so they never
        # collide with a real slot; legal ranks are unique within a row, so no
        # two valid entries target the same slot.
        slot = torch.where(valid, rank, torch.full_like(rank, K))        # [N, A] in [0, K]
        actions_ext = torch.full((n, K + 1), -1, dtype=torch.int32, device=dev)
        actions_ext.scatter_(1, slot, a_idx.to(torch.int32))
        enum_actions = actions_ext[:, :K].contiguous()                   # [N, K] int32, -1 pad
        gather_idx = enum_actions.clamp(min=0).long()
        enum_priors = torch.gather(root_probs, 1, gather_idx)            # [N, K]
        enum_priors = torch.where(
            enum_actions >= 0, enum_priors, torch.zeros_like(enum_priors)
        )
        return enum_actions, enum_priors

    def _initialize_root_from_mask(
        self,
        hidden_batch: torch.Tensor,
        policy_logits_root: torch.Tensor,
        legal_mask: torch.Tensor,
        add_noise: bool,
    ):
        """GPU-resident root setup: legal_mask is already a GPU bool tensor."""
        n = self._allocated_n
        K = self.K
        dev = self.device

        masked_logits = policy_logits_root.masked_fill(~legal_mask, _NEG_INF)
        root_probs = torch.softmax(masked_logits, dim=-1)  # [N, A]

        # Sampled MuZero §5.1: K i.i.d. samples with replacement from β = π,
        # deduped to UNIQUE actions with PUCT prior β̂(a) = count(a)/K on one
        # slot each (padding actions = -1 / prior = 0). Summing an action's mass
        # onto ONE slot (vs Option (b)'s 1/K-per-duplicate-slot) is required so
        # PUCT's prior_score isn't fragmented and swamped by the value term
        # after the first visit.
        sampled_actions = torch.multinomial(root_probs, K, replacement=True)  # [N, K] int64
        actions_dedup, priors_root = self._dedup_sampled(sampled_actions)

        # Per-game guard mirroring the numpy oracle (mcts.py: sample only when
        # ``len(legals) > sample_k``, else expand ALL legal moves with the exact
        # prior). When a game has <= K legal actions (the common case in chess —
        # most positions have ~30 legal moves, K=50), sampling K-with-replacement
        # would waste draws on duplicates, leave some legal moves unexpanded
        # (zero visits -> zero policy-target mass), and replace the exact prior
        # with a noisy count estimate. Expand all legal actions instead. Batched
        # over games of differing legal-move counts via a vectorized select.
        enum_actions, enum_priors = self._enumerate_legal(legal_mask, root_probs)
        use_enum = legal_mask.sum(dim=1, keepdim=True) <= K              # [N, 1] bool
        actions_dedup = torch.where(use_enum, enum_actions, actions_dedup)
        priors_root = torch.where(use_enum, enum_priors, priors_root)

        if add_noise and self.config.dirichlet_epsilon > 0:
            alpha = float(self.config.dirichlet_alpha)
            eps = float(self.config.dirichlet_epsilon)
            # Dirichlet over the UNIQUE actions (matches numpy _add_dirichlet_noise,
            # which draws dir(alpha * len(children)) over the deduped children and
            # blends prior*(1-eps) + noise*eps). Draw K i.i.d. Gamma(alpha) samples,
            # zero them on padding slots, and renormalize per row so the noise is a
            # proper Dirichlet over only the valid (unique) actions.
            valid = actions_dedup != -1                                  # [N, K]
            gamma = torch.distributions.Gamma(
                torch.full((n, K), alpha, device=dev),
                torch.ones((n, K), device=dev),
            ).sample()                                                   # [N, K]
            gamma = torch.where(valid, gamma, torch.zeros_like(gamma))
            noise = gamma / gamma.sum(dim=1, keepdim=True).clamp(min=1e-8)
            priors_root = priors_root * (1.0 - eps) + noise.to(torch.float32) * eps

        sampled_actions = actions_dedup  # name reused by the write below

        # Allocate root at index 0 for every game.
        self.node_count.fill_(1)
        self.node_hidden[:, 0] = hidden_batch.to(self.hidden_dtype)
        # Root reward stays 0; root parent_idx / parent_child_slot stay -1.
        self.child_actions[:, 0, :] = sampled_actions.to(torch.int32)
        self.child_priors[:, 0, :] = priors_root
        # child_visits / child_value_sum / child_rewards / child_node_idx
        # already at sentinel/zero from _reset.

    # -------------------------------------------------------- subtree reuse

    def reset_search(self):
        """Drop any preserved tree state — next ``run_batch_gpu`` does a fresh
        initial_inference and rebuilds the root from scratch."""
        self.has_prior_search = False

    def _refresh_reused_root(self, legal_mask: torch.Tensor, add_noise: bool):
        """Prepare the post-``advance_root`` tree for a new search.

        The new root sits at slot 0 with its sampled children carried over
        from the prior search. Most of those children's actions are legal at
        the new state (they were sampled from the network policy at the time
        the new root was a leaf), but some may be illegal — when the new root
        was previously a non-root leaf, its children were sampled WITHOUT the
        legal-mask filter that's only applied at root expansion. Mask the
        illegal slots out (set ``child_actions = -1``); PUCT skips them. Also
        mix in fresh Dirichlet noise to preserve exploration diversity.
        """
        n = self._allocated_n
        K = self.K
        dev = self.device

        # Mask illegal sampled actions to -1 so PUCT scores them as -inf.
        root_actions = self.child_actions[:, 0, :]                  # [N, K] int32
        valid = root_actions != -1
        # Index legal_mask[g, action]; clamp to safe index for invalid slots.
        action_long = root_actions.clamp(min=0).long()              # [N, K]
        action_legal = torch.gather(legal_mask, 1, action_long) & valid
        self.child_actions[:, 0, :] = torch.where(
            action_legal, root_actions, torch.full_like(root_actions, -1)
        )

        # Mix Dirichlet noise into existing priors. Skip slots we just marked
        # illegal so all noise mass goes to legal slots (re-normalized).
        if add_noise and self.config.dirichlet_epsilon > 0:
            eps = float(self.config.dirichlet_epsilon)
            alpha = float(self.config.dirichlet_alpha)
            noise = torch.distributions.Dirichlet(
                torch.full((K,), alpha, device=dev)
            ).sample((n,)).to(torch.float32)                        # [N, K]
            # Zero noise on illegal slots; renormalize legal noise to sum to 1.
            noise = torch.where(action_legal, noise, torch.zeros_like(noise))
            noise_sum = noise.sum(dim=1, keepdim=True).clamp(min=1e-8)
            noise = noise / noise_sum
            existing = self.child_priors[:, 0, :]
            # Same blend as fresh root: π = (1-ε)·existing + ε·noise.
            self.child_priors[:, 0, :] = existing * (1.0 - eps) + noise * eps

        # MinMaxStats are stale (carry over the prior search's Q range, which
        # may not match the new search's range as it explores). Reset.
        self.mm_min.fill_(float("inf"))
        self.mm_max.fill_(_NEG_INF)

    @torch.no_grad()
    def advance_root(
        self,
        chosen_action: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Promote each game's chosen-action subtree to slot 0 of the tree
        storage. Compacts the tree so subsequent searches start with the
        already-explored subtree as the new root.

        Args:
            chosen_action: [N] int — the action picked at the previous root.
            legal_mask: optional [N, A] bool mask. Currently unused here
                (legal-mask filtering happens in ``_refresh_reused_root``);
                accepted for API symmetry with ``run_batch_gpu``.

        Returns:
            success: [N] bool — True for games where the chosen action's
                child was materialized in the prior search and reuse was
                applied. False games will fall back to a fresh search next
                ply (caller can call ``reset_search`` if any False — but the
                per-game flag is also tracked internally).
        """
        if not (self._allocated_n > 0 and self._allocated_m > 1):
            raise RuntimeError(
                "advance_root called before any run_batch_gpu — no tree state."
            )

        n = self._allocated_n
        m = self._allocated_m
        K = self.K
        dev = self.device
        arange_n = self._arange_n
        arange_m = torch.arange(m, device=dev, dtype=torch.long)

        # 1. Find the slot at root with the chosen action AND a materialized child.
        chosen_i32 = chosen_action.to(torch.int32)
        root_actions = self.child_actions[:, 0, :]                  # [N, K]
        root_child_idx = self.child_node_idx[:, 0, :]               # [N, K]
        match = (root_actions == chosen_i32.unsqueeze(1)) & (root_child_idx != -1)
        # Pick first match per game (argmax of bool → first True or 0 if all False).
        chosen_slot = match.float().argmax(dim=1)                   # [N] int64
        success = match.any(dim=1)                                  # [N] bool

        # 2. New root's old index (per game).
        new_root_old = root_child_idx[arange_n, chosen_slot]        # [N] int32

        # 3. BFS to identify each game's chosen subtree.
        #    in_subtree[g, n] = True iff node n is reachable from new_root_old[g]
        #    via parent pointers (i.e., n is descendant of new_root_old[g]).
        in_subtree = torch.zeros(n, m, dtype=torch.bool, device=dev)
        new_root_l = new_root_old.long().clamp(min=0)
        in_subtree[arange_n, new_root_l] = True
        # Propagate: a node is in_subtree iff its parent is. Tree depth is
        # bounded by MAX_SELECT_DEPTH (search step cap), so that many
        # iterations always converge — no need for a sync-y per-step
        # ``torch.equal`` convergence check.
        parent_l = self.parent_idx.clamp(min=0).long()              # [N, M]
        parent_valid = self.parent_idx != -1
        for _ in range(self.MAX_SELECT_DEPTH + 1):
            parent_in_subtree = in_subtree.gather(1, parent_l)
            in_subtree = in_subtree | (parent_in_subtree & parent_valid)

        # Games where success=False shouldn't reuse. Mark their subtree as
        # empty and we'll handle by setting has_prior_search=False after.
        in_subtree = in_subtree & success.unsqueeze(1)

        # Overflow safety: if any game's subtree + a fresh ply of sims would
        # exceed M, fall back to fresh search. M was sized with the typical
        # case in mind (chosen_subtree ≤ num_sims, fits in 2*num_sims+1).
        # Many consecutive plies of reuse can grow the subtree past num_sims;
        # detect and bail.
        max_subtree_size = int(in_subtree.sum(dim=1).max().item())
        num_sims = int(self.config.num_simulations)
        if max_subtree_size + num_sims > m - 1:
            # Can't fit one more ply of search — drop reuse for this ply.
            self.has_prior_search = False
            return torch.zeros(n, dtype=torch.bool, device=dev)

        # 4. Build permutation [N, M]: for each game, sort node indices so
        #    that (a) new_root_old goes to position 0, then (b) other
        #    in_subtree nodes follow in original order, then (c) out-of-
        #    subtree nodes fill the rest. Sort key:
        #       priority = 0 if n == new_root_old
        #                  1 if in_subtree else 2
        #       tiebreak = original index n
        is_root = arange_m.unsqueeze(0) == new_root_old.long().unsqueeze(1)  # [N, M]
        priority = torch.where(
            is_root,
            torch.zeros_like(arange_m).unsqueeze(0).expand(n, m),
            torch.where(
                in_subtree,
                torch.ones_like(arange_m).unsqueeze(0).expand(n, m),
                torch.full_like(arange_m, 2).unsqueeze(0).expand(n, m),
            ),
        )
        sort_key = priority * m + arange_m.unsqueeze(0)             # [N, M]
        permutation = sort_key.argsort(dim=1)                       # [N, M] int64

        # 5. Inverse permutation: inverse_perm[g, old_idx] = new_idx.
        inverse_perm = torch.empty_like(permutation)
        inverse_perm.scatter_(
            1, permutation, arange_m.unsqueeze(0).expand(n, m).contiguous(),
        )

        # 6. Apply permutation to all per-node tensors via gather along dim 1.
        self.node_visits = torch.gather(self.node_visits, 1, permutation.to(torch.int32).long())
        self.node_value_sum = torch.gather(self.node_value_sum, 1, permutation)
        self.node_reward = torch.gather(self.node_reward, 1, permutation)

        # node_hidden: [N, M, C, H, W]. Broadcast permutation index.
        c, h, w = self._allocated_hidden_shape
        perm_5d = permutation.view(n, m, 1, 1, 1).expand(n, m, c, h, w)
        self.node_hidden = torch.gather(self.node_hidden, 1, perm_5d)

        self.parent_idx = torch.gather(self.parent_idx, 1, permutation.to(torch.int32).long())
        self.parent_child_slot = torch.gather(self.parent_child_slot, 1, permutation.to(torch.int32).long())

        perm_3d = permutation.view(n, m, 1).expand(n, m, K)
        self.child_actions = torch.gather(self.child_actions, 1, perm_3d)
        self.child_priors = torch.gather(self.child_priors, 1, perm_3d)
        self.child_visits = torch.gather(self.child_visits, 1, perm_3d)
        self.child_value_sum = torch.gather(self.child_value_sum, 1, perm_3d)
        self.child_rewards = torch.gather(self.child_rewards, 1, perm_3d)
        child_node_idx_perm = torch.gather(self.child_node_idx, 1, perm_3d)

        # 7. Remap child_node_idx VALUES through inverse_perm. Out-of-subtree
        #    targets (or -1 sentinels) become -1.
        old_idx = child_node_idx_perm                                # [N, M, K] int32
        valid_target = old_idx != -1
        old_idx_safe = old_idx.clamp(min=0).long()                   # [N, M, K]
        new_idx = inverse_perm.gather(1, old_idx_safe.view(n, m * K)).view(n, m, K)
        # Only keep references to nodes that are in the chosen subtree.
        target_in_subtree = in_subtree.gather(1, old_idx_safe.view(n, m * K)).view(n, m, K)
        keep = valid_target & target_in_subtree
        self.child_node_idx = torch.where(
            keep, new_idx.to(torch.int32), torch.full_like(old_idx, -1)
        )

        # 8. Remap parent_idx VALUES. New root (now at slot 0) has parent=-1.
        valid_parent = self.parent_idx != -1
        parent_safe = self.parent_idx.clamp(min=0).long()
        new_parent = inverse_perm.gather(1, parent_safe)
        is_new_root_slot = arange_m.unsqueeze(0) == 0                # [1, M]
        self.parent_idx = torch.where(
            is_new_root_slot,
            torch.full_like(self.parent_idx, -1),
            torch.where(
                valid_parent, new_parent.to(torch.int32),
                torch.full_like(self.parent_idx, -1),
            ),
        )

        # 9. Update node_count to subtree size per game.
        self.node_count = in_subtree.sum(dim=1).to(torch.int32)

        # 10. Clear out-of-subtree slots (positions ≥ subtree size). Reset
        #     child_actions to -1 there so future expansions allocate fresh.
        # The compaction left these slots holding gathered junk from the
        # original positions; stale child_node_idx values pointing to
        # already-remapped positions could mislead future select calls.
        beyond_subtree = arange_m.unsqueeze(0) >= self.node_count.long().unsqueeze(1)  # [N, M]
        beyond_3d = beyond_subtree.unsqueeze(-1)                     # [N, M, 1]
        self.child_actions = torch.where(beyond_3d, torch.full_like(self.child_actions, -1), self.child_actions)
        self.child_priors = torch.where(beyond_3d, torch.zeros_like(self.child_priors), self.child_priors)
        self.child_visits = torch.where(beyond_3d, torch.zeros_like(self.child_visits), self.child_visits)
        self.child_value_sum = torch.where(beyond_3d, torch.zeros_like(self.child_value_sum), self.child_value_sum)
        self.child_rewards = torch.where(beyond_3d, torch.zeros_like(self.child_rewards), self.child_rewards)
        self.child_node_idx = torch.where(beyond_3d, torch.full_like(self.child_node_idx, -1), self.child_node_idx)
        self.node_visits = torch.where(beyond_subtree, torch.zeros_like(self.node_visits), self.node_visits)
        self.node_value_sum = torch.where(beyond_subtree, torch.zeros_like(self.node_value_sum), self.node_value_sum)
        self.node_reward = torch.where(beyond_subtree, torch.zeros_like(self.node_reward), self.node_reward)
        self.parent_idx = torch.where(beyond_subtree, torch.full_like(self.parent_idx, -1), self.parent_idx)
        self.parent_child_slot = torch.where(beyond_subtree, torch.full_like(self.parent_child_slot, -1), self.parent_child_slot)

        # If ANY game's chosen action wasn't materialized, we can't safely
        # reuse — those games would search a stale tree. Take the conservative
        # path: bail out of reuse for the whole batch.
        if not bool(success.all()):
            self.has_prior_search = False
        else:
            self.has_prior_search = True
        return success

    def _select(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized PUCT walk from root to first unexpanded slot.

        Iterates ``MAX_SELECT_DEPTH`` (or ``M-1``, whichever is smaller) PUCT
        steps. The loop body always runs to the bound — no per-step
        ``bool(still_walking.any())`` sync, no compile graph-break. Iterations
        past a game's actual leaf depth are masked out via ``still_walking``
        and write nothing.

        Why a fixed cap (rather than runtime early-exit): in a Sampled MuZero
        tree with K=50 / num_sims=200 the avg depth is ~5-10 and max depth
        rarely exceeds 16-24 (log scaling). Capping the unrolled compile at
        e.g. 32 iterations halves the per-call kernel work vs M-1=200 with
        no measurable correctness impact at our regime — and crucially keeps
        the entire loop in ONE compiled trace, where inductor can fuse 25+
        ops/iter into a few kernels. Adding an early-exit ``bool(...)`` check
        inside the trace causes a graph-break that costs more than the
        wasted iterations save.

        Returns:
            path_node_idx: [N, M] int32 — node visited at each depth (root at
                depth 0); -1 padding past each game's leaf-parent depth.
            path_slot: [N, M] int32 — slot in parent at each depth (-1 at depth 0).
            leaf_parent_node: [N] int32 — node at which selection stopped.
            leaf_slot: [N] int32 — slot in leaf_parent_node selected for the new leaf.
            leaf_action: [N] int32 — action at that slot.
            depth: [N] int32 — index of the leaf_parent in path_node_idx.
        """
        n = self._allocated_n
        m = self._allocated_m
        dev = self.device
        discount = float(self.config.discount)
        arange_n = self._arange_n

        path_node_idx = torch.full((n, m), -1, dtype=torch.int32, device=dev)
        path_slot = torch.full((n, m), -1, dtype=torch.int32, device=dev)
        path_node_idx[:, 0] = 0  # root

        current_node = torch.zeros(n, dtype=torch.int32, device=dev)
        still_walking = torch.ones(n, dtype=torch.bool, device=dev)
        depth = torch.zeros(n, dtype=torch.int32, device=dev)
        leaf_slot = torch.zeros(n, dtype=torch.int32, device=dev)
        leaf_action = torch.zeros(n, dtype=torch.int32, device=dev)

        max_steps = min(m - 1, self.MAX_SELECT_DEPTH)

        for step in range(max_steps):
            cur = current_node.long()
            # Gather current node's child arrays — [N, K] each.
            priors = self.child_priors[arange_n, cur]
            visits = self.child_visits[arange_n, cur]
            value_sums = self.child_value_sum[arange_n, cur]
            rewards = self.child_rewards[arange_n, cur]
            actions = self.child_actions[arange_n, cur]
            node_visits_cur = self.node_visits[arange_n, cur].to(torch.float32)  # [N]

            # PUCT prior score.
            pb_c = (
                torch.log((node_visits_cur + self.PB_C_BASE + 1) / self.PB_C_BASE)
                + self.PB_C_INIT
            )  # [N]
            sqrt_n = torch.sqrt(node_visits_cur)  # [N]
            visits_f = visits.to(torch.float32)  # [N, K]
            prior_score = (
                pb_c.unsqueeze(1) * priors * sqrt_n.unsqueeze(1) / (1.0 + visits_f)
            )

            # Mover-POV Q (matches mcts.py _select_child_loop).
            child_value = torch.where(
                visits_f > 0,
                value_sums / torch.clamp(visits_f, min=1.0),
                torch.zeros_like(value_sums),
            )
            raw_q = rewards - discount * child_value  # [N, K]

            # Per-game min-max normalize.
            has_range = (self.mm_max > self.mm_min).unsqueeze(1)  # [N, 1]
            mm_span = (self.mm_max - self.mm_min).clamp(min=1e-12).unsqueeze(1)
            q_norm = (raw_q - self.mm_min.unsqueeze(1)) / mm_span
            normalized_q = torch.where(has_range, q_norm, raw_q)
            value_score = torch.where(visits_f > 0, normalized_q, torch.zeros_like(normalized_q))

            scores = prior_score + value_score
            # Mask invalid slots so argmax never picks them.
            scores = torch.where(
                actions == -1, torch.full_like(scores, _NEG_INF), scores
            )

            next_slot = scores.argmax(dim=1)  # [N] int64
            next_slot_i32 = next_slot.to(torch.int32)
            chosen_action = actions[arange_n, next_slot]  # [N] int32
            chosen_child = self.child_node_idx[arange_n, cur, next_slot]  # [N] int32

            # A game stops if it was still_walking AND the chosen slot is unexpanded.
            unexpanded = chosen_child == -1
            stops_now = still_walking & unexpanded
            advance = still_walking & ~unexpanded

            # Capture leaf info for games that stop on this step.
            leaf_slot = torch.where(stops_now, next_slot_i32, leaf_slot)
            leaf_action = torch.where(stops_now, chosen_action, leaf_action)
            depth = torch.where(
                stops_now, torch.full_like(depth, step), depth
            )

            # Advance path for games still walking.
            new_step = step + 1
            path_node_idx[:, new_step] = torch.where(
                advance, chosen_child, torch.full_like(chosen_child, -1)
            )
            path_slot[:, new_step] = torch.where(
                advance, next_slot_i32, torch.full_like(next_slot_i32, -1)
            )
            current_node = torch.where(advance, chosen_child, current_node)

            still_walking = advance

        # Safety: if any game is still walking past the cap, force-stop it
        # at the deepest visited node. The next slot it would pick gets
        # treated as the new leaf — wrong if that slot already has a child
        # (its subtree gets orphaned), but the cap is set high enough
        # (typically 32) that this is exceedingly rare in practice. To
        # guarantee correctness, raise ``MAX_SELECT_DEPTH``.
        # We compute leaf info for any game that's still_walking using the
        # final iteration's chosen slot/action.
        force_stop = still_walking
        if max_steps > 0:
            leaf_slot = torch.where(force_stop, next_slot_i32, leaf_slot)
            leaf_action = torch.where(force_stop, chosen_action, leaf_action)
            depth = torch.where(
                force_stop,
                torch.full_like(depth, max_steps - 1),
                depth,
            )

        leaf_parent_node = current_node
        return path_node_idx, path_slot, leaf_parent_node, leaf_slot, leaf_action, depth

    def _select_triton(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Triton-driven PUCT walk: ONE kernel launch handles the full
        depth walk for all N games in parallel.

        The depth loop runs inside the kernel (``tl.static_range``) with all
        per-game state held in registers — no per-step launch, no kernel-
        boundary state shuffling. The safety force-stop is also handled
        inside the kernel.
        """
        n = self._allocated_n
        m = self._allocated_m
        dev = self.device

        path_node_idx = torch.full((n, m), -1, dtype=torch.int32, device=dev)
        path_slot = torch.full((n, m), -1, dtype=torch.int32, device=dev)
        path_node_idx[:, 0] = 0  # root

        leaf_parent_node = torch.zeros(n, dtype=torch.int32, device=dev)
        depth = torch.zeros(n, dtype=torch.int32, device=dev)
        leaf_slot = torch.zeros(n, dtype=torch.int32, device=dev)
        leaf_action = torch.zeros(n, dtype=torch.int32, device=dev)

        max_steps = min(m - 1, self.MAX_SELECT_DEPTH)
        from src.mcts.tensor_mcts_triton import puct_walk
        puct_walk(
            max_steps,
            child_priors=self.child_priors,
            child_visits=self.child_visits,
            child_value_sum=self.child_value_sum,
            child_rewards=self.child_rewards,
            child_actions=self.child_actions,
            child_node_idx=self.child_node_idx,
            node_visits=self.node_visits,
            path_node_idx=path_node_idx,
            path_slot=path_slot,
            leaf_parent_node=leaf_parent_node,
            leaf_slot=leaf_slot,
            leaf_action=leaf_action,
            depth=depth,
            mm_min=self.mm_min,
            mm_max=self.mm_max,
            pb_c_base=self.PB_C_BASE,
            pb_c_init=self.PB_C_INIT,
            discount=float(self.config.discount),
        )

        return path_node_idx, path_slot, leaf_parent_node, leaf_slot, leaf_action, depth

    def _expand(
        self,
        leaf_parent_node: torch.Tensor,
        leaf_slot: torch.Tensor,
        leaf_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate one new node per game; sample K children for it.

        Returns ``(new_node_idx, leaf_value)`` — the new node index per game
        and the leaf network value (used by backprop).
        """
        n = self._allocated_n
        K = self.K
        dev = self.device
        arange_n = self._arange_n

        # Gather parent hidden states.
        parent_l = leaf_parent_node.long()
        parent_hidden = self.node_hidden[arange_n, parent_l]  # [N, C, H, W]

        # Cast to compute dtype before the network call. If storage is fp16
        # and the network is fp32, we'd otherwise hand fp16 into recurrent
        # ops that don't auto-promote. Autocast handles the down-cast inside
        # the network when AMP is enabled.
        compute_dtype = self._compute_dtype()
        with self._amp_ctx():
            next_hidden, rewards, policy_logits, leaf_values = self.network.recurrent_inference(
                parent_hidden.to(compute_dtype),
                leaf_action.long(),
            )

        # Sample K leaf children i.i.d. with replacement from β = π, deduped to
        # UNIQUE actions with PUCT prior β̂(a) = count(a)/K on one slot each
        # (padding = -1 / prior 0). Mirrors the numpy oracle (mcts.py BatchedMCTS
        # leaf expansion). No Dirichlet noise at leaves (root-only).
        leaf_probs = torch.softmax(policy_logits, dim=-1)
        leaf_sampled = torch.multinomial(leaf_probs, K, replacement=True)  # [N, K] int64
        leaf_sampled, leaf_priors = self._dedup_sampled(leaf_sampled)

        # Allocate new node slots.
        new_node_idx = self.node_count.clone()  # [N] int32
        self.node_count = self.node_count + 1
        new_l = new_node_idx.long()
        slot_l = leaf_slot.long()

        # Write new node fields.
        rewards_f32 = rewards.view(-1).to(torch.float32)
        self.node_hidden[arange_n, new_l] = next_hidden.to(self.hidden_dtype)
        self.node_reward[arange_n, new_l] = rewards_f32
        self.parent_idx[arange_n, new_l] = leaf_parent_node
        self.parent_child_slot[arange_n, new_l] = leaf_slot
        self.child_actions[arange_n, new_l, :] = leaf_sampled.to(torch.int32)
        self.child_priors[arange_n, new_l, :] = leaf_priors
        # child_visits / child_value_sum / child_rewards / child_node_idx for
        # the new node remain 0/-1 from _reset.

        # Hook up parent's slot.
        self.child_node_idx[arange_n, parent_l, slot_l] = new_node_idx
        # Mirror reward into parent's child_rewards so PUCT picks it up.
        self.child_rewards[arange_n, parent_l, slot_l] = rewards_f32

        return new_node_idx, leaf_values.view(-1).to(torch.float32)

    def _backprop(
        self,
        path_node_idx: torch.Tensor,
        path_slot: torch.Tensor,
        depth: torch.Tensor,
        new_node_idx: torch.Tensor,
        leaf_slot: torch.Tensor,
        leaf_value: torch.Tensor,
    ):
        """Vectorized backprop: closed-form value-per-depth + parallel scatters.

        Replaces the leaf→root for-loop with a single tensor expression for
        the per-depth values and three scatter_add_'s for the visit/value/
        mirror updates. No Python loop, no per-sim syncs.

        Sign convention (mover-POV): the value added to node_value_sum at depth
        ``d`` (in d's POV) is

            v[d] = node_reward[d+1] − γ·v[d+1],   v[L] = V_leaf

        Closed-form (s = −γ):

            v[d] = (−γ)^(L−d) · V_leaf
                 + Σ_{k=d+1..L} (−γ)^(k−d−1) · node_reward[k]

        Computed via a suffix-cumsum on s^k · node_reward[k] then a prefix
        scaling by s^(−d−1). Per-game leaf depth L[g] = depth[g] + 1.
        """
        n = self._allocated_n
        m = self._allocated_m
        K = self.K
        dev = self.device
        discount = float(self.config.discount)
        arange_n = self._arange_n

        # Extend path with the new leaf node at depth+1.
        new_depth = depth + 1
        new_depth_l = new_depth.long()
        path_node_idx[arange_n, new_depth_l] = new_node_idx
        path_slot[arange_n, new_depth_l] = leaf_slot

        # ---- Path-aligned reward + valid mask --------------------------------
        valid_mask = path_node_idx != -1                            # [N, M]
        path_clamped = path_node_idx.clamp(min=0).long()            # [N, M]
        R_path = torch.gather(self.node_reward, 1, path_clamped)    # [N, M]
        R_path = torch.where(valid_mask, R_path, torch.zeros_like(R_path))

        # ---- s^d powers (no negative-base power op) --------------------------
        # gamma^d via cumprod; sign via parity. Avoids torch ** with neg base.
        arange_d = torch.arange(m, device=dev, dtype=torch.float32)
        arange_d_long = arange_d.long()
        gamma_seed = torch.full((m,), discount, device=dev, dtype=torch.float32)
        gamma_seed[0] = 1.0
        gamma_pow = gamma_seed.cumprod(dim=0)                       # [M] = γ^d
        # (−1)^d → +1 for even d, −1 for odd.
        sign_pow = 1.0 - 2.0 * (arange_d_long % 2).to(torch.float32)  # [M]
        s_pow = sign_pow * gamma_pow                                # [M] = (−γ)^d

        # ---- v[d] closed form ------------------------------------------------
        # u[g, k] = s^k · R_path[g, k];  T[g, d] = Σ_{k≥d} u[g, k] (suffix-cumsum).
        u = s_pow.unsqueeze(0) * R_path                             # [N, M]
        T = u.flip(dims=(1,)).cumsum(dim=1).flip(dims=(1,))         # [N, M]
        # T_shift[g, d] = T[g, d+1] (last col → 0).
        T_shift = torch.cat(
            [T[:, 1:], torch.zeros(n, 1, device=dev, dtype=T.dtype)],
            dim=1,
        )
        # inv_pow[d] = s^(−d−1) = sign_inv * γ^(−d−1).
        # γ^(−d−1) = (1/γ)^(d+1) for γ>0; for γ=1 just 1.
        inv_gamma = (1.0 / discount) if discount != 0.0 else 0.0
        inv_gamma_seed = torch.full((m,), inv_gamma, device=dev, dtype=torch.float32)
        inv_gamma_pow = inv_gamma_seed.cumprod(dim=0)               # [M] = (1/γ)^(d+1)
        sign_inv = -sign_pow                                         # (−1)^(d+1)
        inv_pow = sign_inv * inv_gamma_pow                          # [M]
        first_sum = inv_pow.unsqueeze(0) * T_shift                  # [N, M]

        # leaf_term[g, d] = s^(L[g]−d) · V_leaf for d ≤ L[g], else 0.
        # L[g] = new_depth[g]; only d ≤ L is meaningful (rest masked away).
        L_per_game = new_depth.float().unsqueeze(1)                 # [N, 1]
        leaf_exp = (L_per_game - arange_d.unsqueeze(0)).clamp(min=0)  # [N, M]
        leaf_gamma_pow = discount ** leaf_exp                       # [N, M]
        leaf_sign = 1.0 - 2.0 * (leaf_exp.long() % 2).to(torch.float32)
        leaf_s_pow = leaf_sign * leaf_gamma_pow
        leaf_mask = (arange_d.unsqueeze(0) <= L_per_game).to(torch.float32)
        leaf_term = leaf_s_pow * leaf_value.unsqueeze(1) * leaf_mask  # [N, M]

        value_at_d = first_sum + leaf_term                          # [N, M]
        value_at_d = torch.where(valid_mask, value_at_d, torch.zeros_like(value_at_d))

        # ---- node_visits / node_value_sum scatters ---------------------------
        # path indices within a single game are unique (tree path), so
        # scatter_add along dim=1 has no intra-game duplicates. Invalid entries
        # contribute 0 deltas and harmlessly land on index 0.
        visit_delta = valid_mask.to(self.node_visits.dtype)         # [N, M]
        self.node_visits.scatter_add_(1, path_clamped, visit_delta)
        self.node_value_sum.scatter_add_(1, path_clamped, value_at_d)

        # ---- Mirror to child_visits / child_value_sum ------------------------
        # parent_for_d[g, d] = path[g, d-1] (path's parent node at depth d).
        # Sentinel: parent_for_d[g, 0] = 0; mirror_mask masks d=0 out anyway.
        parent_for_d = torch.cat(
            [
                torch.zeros((n, 1), device=dev, dtype=path_node_idx.dtype),
                path_node_idx[:, :-1],
            ],
            dim=1,
        )
        parent_clamped = parent_for_d.clamp(min=0).long()           # [N, M]
        slots_clamped = path_slot.clamp(min=0).long()               # [N, M]

        d_gt_0 = arange_d_long.unsqueeze(0) > 0                     # [1, M]
        mirror_mask = valid_mask & d_gt_0                           # [N, M]

        # child_visits / child_value_sum mirror node_visits / node_value_sum
        # at child positions. We can apply the SAME deltas to both because the
        # mirror is incremented in lock-step (initial child_visits[*, *, *] = 0
        # at allocation, so the running totals stay in sync).
        child_visit_delta = mirror_mask.to(self.child_visits.dtype)
        child_value_delta = torch.where(
            mirror_mask, value_at_d, torch.zeros_like(value_at_d)
        )

        # Flatten the (M, K) axis to (M*K) so a single dim=1 scatter_add covers
        # the (parent_idx, slot) combination. For invalid d the delta is 0;
        # the destination index is 0 (parent=0, slot=0), so the no-op write
        # accumulates harmlessly with whatever d=1 wrote (additive identity).
        flat_child_idx = parent_clamped * K + slots_clamped         # [N, M]
        self.child_visits.view(n, m * K).scatter_add_(
            1, flat_child_idx, child_visit_delta
        )
        self.child_value_sum.view(n, m * K).scatter_add_(
            1, flat_child_idx, child_value_delta
        )

        # ---- MinMaxStats update ----------------------------------------------
        # q[d] = node.reward − γ · (post_visits_value / post_visit_count).
        # Reduce min/max per game over the path; mask invalid with ±inf so
        # they don't shift the range.
        new_visits_at_d = torch.gather(self.node_visits, 1, path_clamped).to(torch.float32)
        new_value_sum_at_d = torch.gather(self.node_value_sum, 1, path_clamped)
        node_value_at_d = new_value_sum_at_d / torch.clamp(new_visits_at_d, min=1.0)
        q_at_d = R_path - discount * node_value_at_d                # [N, M]

        q_for_min = torch.where(valid_mask, q_at_d, torch.full_like(q_at_d, float("inf")))
        q_for_max = torch.where(valid_mask, q_at_d, torch.full_like(q_at_d, _NEG_INF))
        self.mm_min = torch.minimum(self.mm_min, q_for_min.min(dim=1).values)
        self.mm_max = torch.maximum(self.mm_max, q_for_max.max(dim=1).values)

    # ----------------------------------------------------------- compat shim

    def _build_compat_roots(self, n: int) -> list[MCTSNode]:
        """Copy root data to CPU and build MCTSNode-equivalent objects.

        Aggregates duplicate slots (option (b)) into per-action totals so the
        existing ``select_action`` / ``select_action_gumbel`` / replay-buffer
        plumbing sees one entry per unique sampled action. This is the only
        sync per ply for the MCTS tree.
        """
        # Pack all root-row reads into ONE fp32 tensor for a single .cpu()
        # transfer. Int32 fields (visit_count, child_actions, child_visits)
        # cast to fp32 losslessly because every reachable value is < 2^23
        # (action_space ≤ 4672, max visits ≤ num_simulations).
        K = self.K
        packed = torch.cat(
            [
                self.node_visits[:, 0:1].to(torch.float32),     # [N, 1]
                self.node_value_sum[:, 0:1],                     # [N, 1]
                self.child_actions[:, 0, :].to(torch.float32),   # [N, K]
                self.child_visits[:, 0, :].to(torch.float32),    # [N, K]
                self.child_priors[:, 0, :],                       # [N, K]
                self.child_value_sum[:, 0, :],                    # [N, K]
                self.child_rewards[:, 0, :],                      # [N, K]
            ],
            dim=1,
        )
        cpu = packed.cpu().numpy()                               # ONE sync
        root_visit_count = cpu[:, 0]
        root_value_sum = cpu[:, 1]
        root_actions = cpu[:, 2 : 2 + K]
        root_visits = cpu[:, 2 + K : 2 + 2 * K]
        root_priors = cpu[:, 2 + 2 * K : 2 + 3 * K]
        root_value_sums_per_slot = cpu[:, 2 + 3 * K : 2 + 4 * K]
        root_rewards_per_slot = cpu[:, 2 + 4 * K : 2 + 5 * K]

        roots: list[MCTSNode] = []
        for g in range(n):
            actions_row = root_actions[g]
            valid = actions_row != -1
            actions_v = actions_row[valid]
            visits_v = root_visits[g][valid].astype(np.float64)
            priors_v = root_priors[g][valid].astype(np.float64)
            value_sums_v = root_value_sums_per_slot[g][valid].astype(np.float64)
            rewards_v = root_rewards_per_slot[g][valid].astype(np.float64)

            # Aggregate by unique action (collapse option-(b) duplicates).
            if actions_v.size == 0:
                # Pathological — terminal state at root. Caller shouldn't hit this.
                node = MCTSNode(visit_count=0, value_sum=0.0)
                node.child_actions = np.zeros(0, dtype=np.int64)
                node.child_visits = np.zeros(0, dtype=np.float64)
                node.child_priors = np.zeros(0, dtype=np.float64)
                node.child_value_sums = np.zeros(0, dtype=np.float64)
                node.child_rewards = np.zeros(0, dtype=np.float64)
                node.children = []
                roots.append(node)
                continue

            unique, inverse = np.unique(actions_v, return_inverse=True)
            agg_visits = np.zeros(len(unique), dtype=np.float64)
            agg_priors = np.zeros(len(unique), dtype=np.float64)
            agg_value_sums = np.zeros(len(unique), dtype=np.float64)
            agg_rewards = np.zeros(len(unique), dtype=np.float64)
            count_per_action = np.zeros(len(unique), dtype=np.float64)
            np.add.at(agg_visits, inverse, visits_v)
            np.add.at(agg_priors, inverse, priors_v)
            np.add.at(agg_value_sums, inverse, value_sums_v)
            np.add.at(count_per_action, inverse, np.ones_like(visits_v))
            # child_rewards is constant per action (deterministic dynamics);
            # take the mean over duplicate slots.
            np.add.at(agg_rewards, inverse, rewards_v)
            agg_rewards = agg_rewards / np.clip(count_per_action, 1.0, None)

            node = MCTSNode(
                visit_count=int(root_visit_count[g]),
                value_sum=float(root_value_sum[g]),
            )
            node.child_actions = unique.astype(np.int64)
            node.child_visits = agg_visits
            node.child_priors = agg_priors
            node.child_value_sums = agg_value_sums
            node.child_rewards = agg_rewards
            # Lazy children list — present so .expanded() returns True.
            node.children = [None] * len(unique)
            roots.append(node)

        return roots
