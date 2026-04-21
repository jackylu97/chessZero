"""Monte Carlo Tree Search for MuZero."""

import math
from dataclasses import dataclass

import numpy as np
import torch

from src.mcts.gumbel import (
    compute_improved_policy,
    compute_v_mix,
    sequential_halving_schedule,
)
from src.mcts.sampling import sample_topk_gumbel


class MinMaxStats:
    """Tracks min/max values seen during search for Q-value normalization."""

    def __init__(self):
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, value: float):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MCTSNode:
    """A node in the MCTS search tree.

    Parallel-array storage (`child_priors`, `child_visits`, `child_value_sums`,
    `child_rewards`) is the source of truth for child stats during selection.
    The `children` list is lazily populated: slots start as `None` and are
    instantiated only when a child is actually traversed — cuts allocations
    by ~5-17× vs eager expansion at Sampled-MuZero-scale K.

    `__slots__` + hand-rolled `__init__` makes construction ~3× faster than
    the dataclass-generated version (eliminates `__dict__`, skips per-kwarg
    default-handling overhead).
    """
    __slots__ = (
        "hidden_state", "prior", "reward", "visit_count", "value_sum",
        "children",
        "child_actions", "child_priors", "child_visits",
        "child_rewards", "child_value_sums",
        "value_prior", "root_legal_actions", "root_legal_logits",
        "root_sampled_perturbed", "min_max_q",
    )

    def __init__(self, prior: float = 0.0, visit_count: int = 0,
                 value_sum: float = 0.0, reward: float = 0.0):
        self.prior = prior
        self.visit_count = visit_count
        self.value_sum = value_sum
        self.reward = reward
        self.hidden_state = None
        self.children = []  # lazily filled with None placeholders after expansion
        self.child_actions = None       # (K,) int64
        self.child_priors = None        # (K,) float64 — PUCT prior over σ. Under Sampled MuZero (Hubert 2021 Proposed Modification) this is π_net renormalized to sum to 1 over σ; Dirichlet noise mutates it at the root.
        self.child_visits = None        # (K,) float64
        self.child_rewards = None       # (K,) float64
        self.child_value_sums = None    # (K,) float64
        # Plain Gumbel MuZero (Danihelka 2022) — populated at the root only.
        self.value_prior = None              # v̂_π for v_mix
        self.root_legal_actions = None       # (|legals|,) int64
        self.root_legal_logits = None        # (|legals|,) float64 — raw logits at legal actions; used to build dense π' training target
        self.root_sampled_perturbed = None   # (m,) float64 — g(a)+logits(a) at sampled actions, parallel to child_actions
        self.min_max_q = None                # (min, max) Q bounds after search

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """PUCT-based MCTS for MuZero."""

    PB_C_BASE = 19652
    PB_C_INIT = 1.25

    def __init__(
        self,
        network,
        game,
        config,
        device: str = "cpu",
    ):
        self.network = network
        self.game = game
        self.config = config
        self.device = device
        # RNG for Gumbel-Top-K sampling at root. Leaves use torch RNG inline on GPU.
        self._rng = np.random.default_rng()

    @torch.no_grad()
    def run(
        self,
        observation: torch.Tensor,
        legal_actions: list[int],
        add_noise: bool = True,
    ) -> MCTSNode:
        """Run MCTS simulations from root observation."""
        root = MCTSNode()
        min_max_stats = MinMaxStats()

        obs_batch = observation.unsqueeze(0).to(self.device)
        hidden, policy_logits, value = self.network.initial_inference(obs_batch)

        root.hidden_state = hidden.squeeze(0)
        self._expand(root, policy_logits.squeeze(0), legal_actions)

        if add_noise and len(root.children) > 0:
            self._add_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            path_indices: list[int] = [-1]  # root has no parent index; sentinel

            while node.expanded():
                action, idx, child = self._select_child(node, min_max_stats)
                node = child
                search_path.append(node)
                path_indices.append(idx)

            parent = search_path[-2]
            parent_hidden = parent.hidden_state.unsqueeze(0).to(self.device)
            action_tensor = torch.tensor([action], device=self.device)
            next_hidden, reward, policy_logits, value = self.network.recurrent_inference(
                parent_hidden, action_tensor
            )
            node.hidden_state = next_hidden.squeeze(0)
            node.reward = reward.item()

            all_actions = list(range(self.game.action_space_size))
            self._expand(node, policy_logits.squeeze(0), all_actions)

            self._backpropagate(search_path, path_indices, value.item(), min_max_stats)

        return root

    def _add_dirichlet_noise(self, node: MCTSNode):
        """Mix Dirichlet noise into a node's child priors (applied only at root).

        Selection reads priors from `child_priors` directly, so there is no
        per-child attribute to sync. A child that gets materialized *after*
        this call picks up the noised prior via `_get_or_create_child`.
        """
        k = len(node.children)
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * k)
        eps = self.config.dirichlet_epsilon
        node.child_priors = node.child_priors * (1.0 - eps) + noise * eps
        # Propagate noise to children that were already materialized, so the
        # child.prior attribute (used only as the seed for lazy materialization
        # and readable externally) stays consistent with the parent array.
        for i, child in enumerate(node.children):
            if child is not None:
                child.prior = float(node.child_priors[i])

    def _expand(self, node: MCTSNode, policy_logits: torch.Tensor,
                legal_actions: list[int]):
        """Expand node with children, populating the parallel-array storage.

        Does its own GPU softmax + CPU transfer. `BatchedMCTS.run_batch` bypasses
        this for per-sim leaves — it batches softmax across all N games into one
        transfer, then calls `_expand_from_priors` below. Leaf sampling (Sampled
        MuZero) is only implemented in the batched path.
        """
        policy = torch.full_like(policy_logits, float("-inf"))
        legal_idx = torch.tensor(legal_actions, dtype=torch.long)
        policy[legal_idx] = policy_logits[legal_idx]
        policy = torch.softmax(policy, dim=0)

        actions_np = np.asarray(legal_actions, dtype=np.int64)
        priors_np = policy[torch.from_numpy(actions_np)].detach().cpu().numpy().astype(np.float64)

        self._expand_from_priors(node, actions_np, priors_np)

    def _expand_from_priors(self, node: MCTSNode,
                            actions_np: np.ndarray, priors_np: np.ndarray):
        """Populate parallel-array storage from pre-computed numpy priors.

        Pure numpy — no GPU ops, no CPU↔GPU transfer. Callers batch the softmax
        and .cpu() transfer across all games in a sim step, then hand off here.

        `priors_np` is the PUCT prior over the children being installed. For
        Sampled MuZero the sampled call sites renormalize π_net over σ before
        calling this; for full expansion it is the full masked softmax. Root
        Dirichlet noise is applied later via `_add_dirichlet_noise` on
        `child_priors`.

        Children are allocated lazily: `node.children` is a list of `None`
        placeholders. `_select_child_*` materializes the concrete MCTSNode
        the first time a slot is traversed. For Sampled MuZero with K=50 and
        100 simulations, PUCT typically touches ~20-30 of the 50 slots, so
        this cuts per-expansion allocation by ~2×; for full-action expansion
        it cuts it by ~(action_space / sims).
        """
        k = len(actions_np)
        node.child_actions = actions_np
        node.child_priors = np.array(priors_np, dtype=np.float64)
        node.child_visits = np.zeros(k, dtype=np.float64)
        node.child_rewards = np.zeros(k, dtype=np.float64)
        node.child_value_sums = np.zeros(k, dtype=np.float64)
        node.children = [None] * k

    def _get_or_create_child(self, node: MCTSNode, idx: int) -> MCTSNode:
        """Materialize a lazy child slot. Safe to call when the slot is already set."""
        child = node.children[idx]
        if child is None:
            child = MCTSNode(prior=float(node.child_priors[idx]))
            node.children[idx] = child
        return child

    # Dispatch threshold: numpy has ~7us fixed overhead per call; the Python loop
    # (reading attrs directly off child objects) wins through K~50. Measured
    # crossover: loop 7.35us / numpy 7.45us at K=50; loop 8.64us / numpy 7.47us
    # at K=60. Threshold chosen so both paths are used where each is clearly faster.
    _VECTORIZE_THRESHOLD = 60

    def _select_child(self, node: MCTSNode, min_max_stats: MinMaxStats) -> tuple[int, int, MCTSNode]:
        """PUCT selection. Returns (action, child_index, child_node).

        `child_index` is the position in the parent's parallel arrays —
        `_backpropagate` needs it to sync updated child stats back into the parent.
        """
        k = len(node.children)
        if k >= self._VECTORIZE_THRESHOLD:
            return self._select_child_vectorized(node, min_max_stats)
        return self._select_child_loop(node, min_max_stats)

    def _select_child_loop(self, node: MCTSNode, min_max_stats: MinMaxStats) -> tuple[int, int, MCTSNode]:
        """Python-loop PUCT — faster than numpy for small child counts (tictactoe/connect4).

        Reads stats from the parent's parallel arrays (source of truth under
        lazy child allocation). Materializes the selected child on return.
        Arrays are tolist()'d once up-front so per-iteration reads hit Python
        list objects, not numpy scalars (~40ns vs ~80ns).
        """
        pb_c = math.log((node.visit_count + self.PB_C_BASE + 1) / self.PB_C_BASE) + self.PB_C_INIT
        sqrt_n = math.sqrt(node.visit_count)
        discount = self.config.discount
        mm_min, mm_max = min_max_stats.minimum, min_max_stats.maximum
        has_range = mm_max > mm_min
        mm_span = mm_max - mm_min if has_range else 1.0
        priors = node.child_priors.tolist()
        visits = node.child_visits.tolist()
        rewards = node.child_rewards.tolist()
        value_sums = node.child_value_sums.tolist()

        best_score = -float("inf")
        best_idx = -1
        for i in range(len(priors)):
            v = visits[i]
            prior_score = pb_c * priors[i] * sqrt_n / (1.0 + v)
            if v > 0:
                raw_q = -rewards[i] - discount * (value_sums[i] / v)
                value_score = (raw_q - mm_min) / mm_span if has_range else raw_q
            else:
                value_score = 0.0
            score = prior_score + value_score
            if score > best_score:
                best_score = score
                best_idx = i
        return int(node.child_actions[best_idx]), best_idx, self._get_or_create_child(node, best_idx)

    def _select_child_vectorized(self, node: MCTSNode, min_max_stats: MinMaxStats) -> tuple[int, int, MCTSNode]:
        """Numpy-vectorized PUCT — faster for large child counts (chess leaves)."""
        priors = node.child_priors
        visits = node.child_visits
        rewards = node.child_rewards
        value_sums = node.child_value_sums

        pb_c = math.log((node.visit_count + self.PB_C_BASE + 1) / self.PB_C_BASE) + self.PB_C_INIT
        sqrt_n = math.sqrt(node.visit_count)

        prior_scores = pb_c * priors * sqrt_n / (1.0 + visits)

        with np.errstate(invalid="ignore", divide="ignore"):
            child_values = np.where(visits > 0, value_sums / np.maximum(visits, 1.0), 0.0)
        raw_q = -rewards - self.config.discount * child_values

        if min_max_stats.maximum > min_max_stats.minimum:
            normalized_q = (raw_q - min_max_stats.minimum) / (min_max_stats.maximum - min_max_stats.minimum)
        else:
            normalized_q = raw_q
        value_scores = np.where(visits > 0, normalized_q, 0.0)

        scores = prior_scores + value_scores
        best_idx = int(np.argmax(scores))
        return int(node.child_actions[best_idx]), best_idx, self._get_or_create_child(node, best_idx)

    def _backpropagate(self, search_path: list[MCTSNode], path_indices: list[int],
                       value: float, min_max_stats: MinMaxStats):
        """Backprop the leaf value up the search path and sync parent arrays.

        path_indices[i] is the index of search_path[i] in search_path[i-1]'s
        child arrays (or -1 for the root, which has no parent).
        """
        n = len(search_path)
        for depth in range(n - 1, -1, -1):
            node = search_path[depth]
            sign = 1.0 if ((n - 1 - depth) % 2 == 0) else -1.0
            node.visit_count += 1
            node.value_sum += value * sign

            if depth > 0:
                parent = search_path[depth - 1]
                idx = path_indices[depth]
                parent.child_visits[idx] = node.visit_count
                parent.child_value_sums[idx] = node.value_sum
                parent.child_rewards[idx] = node.reward

            if node.visit_count > 0:
                min_max_stats.update(-node.reward - self.config.discount * node.value)

            value = node.reward + self.config.discount * value


def _masked_softmax_np(logits: np.ndarray, legal_actions: np.ndarray) -> np.ndarray:
    """Softmax over `legal_actions` only. Returns probs of shape (len(legal_actions),)."""
    x = logits[legal_actions]
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


@dataclass
class _GumbelRootState:
    """Per-game state for Plain Gumbel MuZero Sequential Halving at the root.

    Lives for one MCTS run. Indices stored are *local* positions in the root's
    child arrays (0..m-1 after the Gumbel-Top-K sample).
    """
    m: int
    schedule: list  # list[(m_p, vpa)] from sequential_halving_schedule
    phase: int = 0
    surviving: np.ndarray | None = None   # (m_p,) local indices still alive
    queue: list | None = None             # local indices to visit in current phase
    queue_pos: int = 0


class BatchedMCTS(MCTS):
    """MCTS over N games simultaneously, batching all network calls.

    Sync budget per simulation: **3 GPU→CPU transfers** regardless of N —
    one bulk `.tolist()` for rewards, one for values, one `.cpu()` for
    policy indices+priors. The previous implementation did 3N transfers
    (one per game per step); at N=64 this is a ~60× reduction in
    sync-barrier count.

    If `config.sample_k` is set, uses Gumbel-Top-K sampling (Kool 2019) at
    roots and leaves — the Sampled MuZero policy-improvement operator
    (Hubert 2021, Proposed Modification). After sampling σ, priors are
    renormalized over σ so PUCT uses π_net restricted to the sampled
    subspace; the training target is then raw `N(a)/ΣN` — Theorem 1 gives
    that this already approximates the improved policy Î_β π. No post-hoc
    β̂/β correction at target time (the Naive variant the paper flags as
    unstable). `sample_k=None` expands the full action space at each leaf
    (tractable only for small action spaces).

    Under ``config.use_gumbel`` (Plain Gumbel MuZero, Danihelka 2022): the
    root's PUCT + Dirichlet path is replaced by Sequential Halving on
    ``gumbel_num_considered`` actions sampled via Gumbel-Top-K; non-root
    selection stays PUCT. Leaf sampling (``sample_k``) is independent and
    still applies to chess-size action spaces.
    """

    # --- Gumbel root Sequential Halving helpers -----------------------------
    # Methods below operate on a _GumbelRootState and the root MCTSNode's
    # sampled children. Pure numpy; called once per simulation per game.

    def _gumbel_build_phase_queue(self, state: _GumbelRootState, vpa: int):
        """Round-robin queue: each surviving index visited `vpa` times."""
        state.queue = list(state.surviving) * int(vpa)
        state.queue_pos = 0

    def _gumbel_score(self, root: MCTSNode, local_indices: np.ndarray,
                      mm: MinMaxStats) -> np.ndarray:
        """Paper Eq 8 comparator: g(a) + logits(a) + σ(q̂(a)).

        ``root_sampled_perturbed`` already holds g+logits at sampled indices;
        σ(q̂) is computed here from min/max-normalized Q.
        """
        perturbed = root.root_sampled_perturbed[local_indices]
        visits = root.child_visits[local_indices]
        value_sums = root.child_value_sums[local_indices]
        rewards = root.child_rewards[local_indices]
        with np.errstate(invalid="ignore", divide="ignore"):
            child_values = np.where(
                visits > 0, value_sums / np.maximum(visits, 1.0), 0.0
            )
        raw_q = -rewards - self.config.discount * child_values
        if mm.maximum > mm.minimum:
            q_norm = (raw_q - mm.minimum) / (mm.maximum - mm.minimum)
        else:
            q_norm = raw_q
        q_norm = np.where(visits > 0, q_norm, 0.5)
        max_n = float(root.child_visits.max()) if root.child_visits.size > 0 else 0.0
        sig = (self.config.gumbel_c_visit + max_n) * self.config.gumbel_c_scale * q_norm
        return perturbed + sig

    def _gumbel_advance_phase(self, state: _GumbelRootState, root: MCTSNode,
                              mm: MinMaxStats) -> bool:
        """Rank surviving by score, keep top half, build next phase queue."""
        state.phase += 1
        if state.phase >= len(state.schedule):
            return False
        m_p_new, vpa = state.schedule[state.phase]
        scores = self._gumbel_score(root, state.surviving, mm)
        keep = min(int(m_p_new), len(state.surviving))
        order = np.argsort(-scores)
        state.surviving = state.surviving[order[:keep]]
        self._gumbel_build_phase_queue(state, vpa)
        return True

    def _gumbel_next_root_index(self, state: _GumbelRootState, root: MCTSNode,
                                 mm: MinMaxStats) -> int:
        """Next local child index to visit at the root under Sequential Halving.

        Pops from the current phase queue; advances phases when drained. If
        the schedule exhausts before ``num_simulations`` (rounding-loss
        overhang), returns the provisional best by g+logits+σ(q̂).
        """
        while state.queue_pos >= len(state.queue):
            if not self._gumbel_advance_phase(state, root, mm):
                scores = self._gumbel_score(root, state.surviving, mm)
                return int(state.surviving[int(np.argmax(scores))])
        idx = int(state.queue[state.queue_pos])
        state.queue_pos += 1
        return idx

    @torch.no_grad()
    def run_batch(
        self,
        observations: list[torch.Tensor],
        legal_actions_list: list[list[int]],
        add_noise: bool = True,
    ) -> list[MCTSNode]:
        n = len(observations)
        roots = [MCTSNode() for _ in range(n)]
        min_max_stats = [MinMaxStats() for _ in range(n)]

        action_space_size = self.game.action_space_size
        sample_k = getattr(self.config, "sample_k", None)
        use_sampled_leaf = sample_k is not None and sample_k < action_space_size
        use_gumbel_root = bool(getattr(self.config, "use_gumbel", False))
        gumbel_noise_on = (
            use_gumbel_root
            and add_noise
            and bool(getattr(self.config, "use_gumbel_noise", True))
        )
        all_actions_np = np.arange(action_space_size, dtype=np.int64)

        obs_batch = torch.stack(observations).to(self.device)
        hidden_batch_gpu, policy_batch, value_batch = \
            self.network.initial_inference(obs_batch)

        # One bulk transfer of all root logits; mask+softmax per game in numpy.
        root_logits_cpu = policy_batch.detach().cpu().numpy().astype(np.float64)
        root_values_cpu = value_batch.view(-1).tolist() if use_gumbel_root else None

        gumbel_states: list[_GumbelRootState | None] = [None] * n

        for g in range(n):
            roots[g].hidden_state = hidden_batch_gpu[g]
            legals_np = np.asarray(legal_actions_list[g], dtype=np.int64)
            legal_logits = root_logits_cpu[g][legals_np]
            full_priors = _masked_softmax_np(root_logits_cpu[g], legals_np)

            if use_gumbel_root:
                # Capture v̂_π and raw legal logits for the π' training target (G3).
                roots[g].value_prior = float(root_values_cpu[g])
                roots[g].root_legal_actions = legals_np
                roots[g].root_legal_logits = legal_logits.copy()

                m = min(int(self.config.gumbel_num_considered), len(legals_np))
                if gumbel_noise_on:
                    sampled_local, perturbed_sampled = sample_topk_gumbel(
                        legal_logits, m, rng=self._rng
                    )
                else:
                    # Eval / noise-off: g ≡ 0, top-m by raw logits (paper Fig 8b).
                    order = np.argsort(-legal_logits)[:m]
                    sampled_local = order.astype(np.int64)
                    perturbed_sampled = legal_logits[sampled_local].copy()

                actions_np = legals_np[sampled_local]
                priors_np = full_priors[sampled_local].copy()
                s = priors_np.sum()
                priors_np = priors_np / s if s > 0 else priors_np
                self._expand_from_priors(roots[g], actions_np, priors_np)
                roots[g].root_sampled_perturbed = perturbed_sampled

                schedule = sequential_halving_schedule(self.config.num_simulations, m)
                state = _GumbelRootState(
                    m=m, schedule=schedule, phase=0,
                    surviving=np.arange(m, dtype=np.int64),
                )
                m_p0, vpa0 = schedule[0]
                state.surviving = state.surviving[:m_p0]
                self._gumbel_build_phase_queue(state, vpa0)
                gumbel_states[g] = state
                # No Dirichlet under Gumbel — Gumbel-Top-K provides exploration.
                continue

            if use_sampled_leaf and len(legals_np) > sample_k:
                # Sample K distinct legal actions via Gumbel-Top-K over their logits.
                # Renormalize π_net over σ so PUCT has a proper prior on the
                # sampled subspace (Hubert 2021 Proposed Modification).
                sampled_local, _ = sample_topk_gumbel(legal_logits, sample_k, rng=self._rng)
                actions_np = legals_np[sampled_local]
                sampled_priors = full_priors[sampled_local]
                priors_np = sampled_priors / sampled_priors.sum()
            else:
                actions_np = legals_np
                priors_np = full_priors
            self._expand_from_priors(roots[g], actions_np, priors_np)
            if add_noise and roots[g].children:
                self._add_dirichlet_noise(roots[g])

        for _ in range(self.config.num_simulations):
            search_paths = []
            path_indices_all = []
            parent_hiddens = []
            leaf_actions = []

            for g in range(n):
                node = roots[g]
                path = [node]
                path_indices = [-1]
                last_action = None

                # Root step: Sequential Halving under Gumbel; PUCT otherwise.
                if gumbel_states[g] is not None:
                    local_idx = self._gumbel_next_root_index(
                        gumbel_states[g], roots[g], min_max_stats[g]
                    )
                    last_action = int(roots[g].child_actions[local_idx])
                    node = self._get_or_create_child(roots[g], local_idx)
                    path.append(node)
                    path_indices.append(local_idx)

                # Non-root steps (and root under PUCT): standard PUCT.
                while node.expanded():
                    last_action, idx, child = self._select_child(node, min_max_stats[g])
                    node = child
                    path.append(node)
                    path_indices.append(idx)

                search_paths.append(path)
                path_indices_all.append(path_indices)
                parent_hiddens.append(path[-2].hidden_state)
                leaf_actions.append(last_action)

            hidden_batch = torch.stack(parent_hiddens).to(self.device)
            action_batch = torch.tensor(leaf_actions, device=self.device)
            next_hiddens, rewards, policy_batch, value_batch = \
                self.network.recurrent_inference(hidden_batch, action_batch)

            # Batched GPU→CPU transfers: 3 syncs for the whole step, not 3×N.
            rewards_list = rewards.view(-1).tolist()
            values_list = value_batch.view(-1).tolist()
            probs_gpu = torch.softmax(policy_batch, dim=-1)
            if use_sampled_leaf:
                # Gumbel-Top-K: argmax(logits + Gumbel) = sample K distinct actions
                # from softmax(logits) without replacement (Kool 2019). Priors
                # are renormalized over σ (Hubert 2021 Proposed Modification).
                u = torch.rand_like(policy_batch).clamp_(1e-20, 1 - 1e-20)
                gumbel = -torch.log(-torch.log(u))
                perturbed = policy_batch + gumbel
                sampled_idx = torch.topk(perturbed, sample_k, dim=-1).indices
                sampled_priors = torch.gather(probs_gpu, dim=-1, index=sampled_idx)
                sampled_priors = sampled_priors / sampled_priors.sum(dim=-1, keepdim=True)
                topk_actions_np = sampled_idx.detach().cpu().numpy().astype(np.int64)
                topk_priors_np = sampled_priors.detach().cpu().numpy().astype(np.float64)
            else:
                probs_np = probs_gpu.detach().cpu().numpy().astype(np.float64)

            for g in range(n):
                leaf = search_paths[g][-1]
                leaf.hidden_state = next_hiddens[g]
                leaf.reward = rewards_list[g]
                if use_sampled_leaf:
                    self._expand_from_priors(leaf, topk_actions_np[g], topk_priors_np[g])
                else:
                    self._expand_from_priors(leaf, all_actions_np, probs_np[g])
                self._backpropagate(search_paths[g], path_indices_all[g],
                                    values_list[g], min_max_stats[g])

        # Capture the post-search Q range for π' completedQ normalization.
        if use_gumbel_root:
            for g in range(n):
                mm = min_max_stats[g]
                roots[g].min_max_q = (float(mm.minimum), float(mm.maximum))

        return roots


def select_action(
    root: MCTSNode,
    temperature: float = 1.0,
) -> tuple[int, np.ndarray]:
    """Select action from MCTS visit counts using temperature.

    The returned dense policy is raw-normalized visit counts `N(a)/ΣN(a)` over
    the root's children. Under Sampled MuZero (Hubert 2021 Proposed
    Modification) the search already accounts for the sampling via the
    renormalized-over-σ prior inside PUCT, so no post-hoc β̂/β correction is
    applied here — Theorem 1 gives that raw visit counts already approximate
    the improved policy Î_β π.

    Returns:
        action: selected action index
        action_probs: dense policy distribution over [0, max(actions)+1)
    """
    actions = root.child_actions
    visits = root.child_visits.astype(np.float64)

    if temperature == 0:
        action = int(actions[int(np.argmax(visits))])
        probs = np.zeros(len(actions))
        probs[int(np.argmax(visits))] = 1.0
    else:
        visits_temp = visits ** (1.0 / temperature)
        total_v = visits_temp.sum()
        if total_v <= 0:
            visit_probs = np.ones(len(actions)) / len(actions)
        else:
            visit_probs = visits_temp / total_v
        action = int(np.random.choice(actions, p=visit_probs))
        probs = visit_probs

    max_action = int(actions.max()) + 1
    action_probs = np.zeros(max_action, dtype=np.float32)
    action_probs[actions] = probs
    return action, action_probs


def select_action_gumbel(
    root: MCTSNode,
    config,
    action_space_size: int,
) -> tuple[int, np.ndarray]:
    """Plain Gumbel MuZero action selection + π' training target (Danihelka 2022).

    Action A_{n+1} (paper page 18): argmax over the m sampled children of
    ``g(a) + logits(a) + σ(q̂(a))``. With ``use_gumbel_noise=False`` (eval /
    reanalyze) ``g ≡ 0`` and this degenerates to argmax over
    ``logits + σ(q̂)`` — deterministic greedy.

    Training target (paper Eq. 11): ``π' = softmax(logits + σ(completedQ))``
    over **all legal actions at root**, with unsampled-action Q filled by
    ``v_mix`` (paper Eq. 33). Q-values are min/max-normalized to [0, 1] using
    the post-search range captured on ``root.min_max_q``; σ uses
    ``max_b N(b)`` across sampled children.

    Returns:
        action: A_{n+1}
        action_probs: dense π' of length ``action_space_size`` (0 for illegal).
    """
    legals = root.root_legal_actions
    legal_logits = root.root_legal_logits
    sampled_actions = root.child_actions
    sampled_visits = root.child_visits.astype(np.float64)
    sampled_rewards = root.child_rewards.astype(np.float64)
    sampled_value_sums = root.child_value_sums.astype(np.float64)
    perturbed = root.root_sampled_perturbed

    # Raw Q at each sampled child (perspective flip + reward — matches PUCT).
    with np.errstate(invalid="ignore", divide="ignore"):
        sampled_child_values = np.where(
            sampled_visits > 0,
            sampled_value_sums / np.maximum(sampled_visits, 1.0),
            0.0,
        )
    raw_q_sampled = -sampled_rewards - config.discount * sampled_child_values

    # Softmax over legal logits — used as π in v_mix.
    shifted = legal_logits - legal_logits.max()
    exp_legal = np.exp(shifted)
    legal_priors = exp_legal / exp_legal.sum()

    L = len(legals)
    action_to_legal_idx = {int(a): i for i, a in enumerate(legals)}
    sampled_legal_pos = np.array(
        [action_to_legal_idx[int(a)] for a in sampled_actions],
        dtype=np.int64,
    )

    # Spread sampled raw-Q and visits back onto the full-legal axis (zeros elsewhere).
    visits_all = np.zeros(L, dtype=np.float64)
    visits_all[sampled_legal_pos] = sampled_visits
    q_raw_all = np.zeros(L, dtype=np.float64)
    q_raw_all[sampled_legal_pos] = raw_q_sampled

    v_prior = float(root.value_prior) if root.value_prior is not None else 0.0
    v_mix_raw = compute_v_mix(v_prior, legal_priors, visits_all, q_raw_all)

    # completedQ in raw-Q space: q(a) if sampled+visited, else v_mix.
    completed_q_raw = np.where(visits_all > 0, q_raw_all, v_mix_raw)

    # Normalize to [0, 1] via the MinMaxStats range captured post-search.
    if root.min_max_q is not None:
        mm_min, mm_max = root.min_max_q
    else:
        mm_min, mm_max = float("inf"), float("-inf")
    if mm_max > mm_min:
        completed_q_norm = (completed_q_raw - mm_min) / (mm_max - mm_min)
    else:
        completed_q_norm = np.clip(completed_q_raw, 0.0, 1.0)

    max_n = float(sampled_visits.max()) if sampled_visits.size > 0 else 0.0
    legal_pi_prime = compute_improved_policy(
        legal_logits,
        completed_q_norm,
        max_n,
        c_visit=config.gumbel_c_visit,
        c_scale=config.gumbel_c_scale,
    )

    dense = np.zeros(action_space_size, dtype=np.float32)
    dense[legals] = legal_pi_prime.astype(np.float32)

    # A_{n+1}: argmax over sampled children of (perturbed + σ(q̂_norm)).
    sampled_q_norm = completed_q_norm[sampled_legal_pos]
    sigma_sampled = (config.gumbel_c_visit + max_n) * config.gumbel_c_scale * sampled_q_norm
    scores_sampled = perturbed + sigma_sampled
    winner_local = int(np.argmax(scores_sampled))
    action = int(sampled_actions[winner_local])

    return action, dense
