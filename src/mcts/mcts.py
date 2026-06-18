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
        # Root-terminal-draws override: (K,) float with NaN=normal, else the
        # mover-POV draw value used DIRECTLY as raw_q in selection (so a winning
        # side avoids the repeating move, a losing side keeps it). terminal_value
        # marks a materialized child as a non-expanding terminal-draw leaf.
        "child_terminal_value", "terminal_value",
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
        self.child_terminal_value = None     # (K,) float64 NaN=normal; else forced-draw mover-POV value
        self.terminal_value = None           # float on a materialized forced-draw child (don't expand)

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

    def _sample_iid_with_replacement(
        self, probs: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hubert 2021 §5.1 / §5.4 sampling step for the Proposed Modification.

        Sample k actions i.i.d. with replacement from β = π. Return
        ``(unique_indices, beta_hat)`` where ``beta_hat[i] = count(unique_i) / k``.

        Under the Proposed Modification with β = π, the PUCT prior π̂_β(a)
        ∝ (β̂(a)/β(a))·π(a) reduces to β̂(a). So β̂ is used directly as the
        PUCT prior over the unique sampled actions. K_unique ≤ k and varies by
        position: for sharp π only a handful of distinct actions sample;
        for diffuse π the sampling fills out close to k distinct actions.
        """
        sampled = self._rng.choice(len(probs), size=k, replace=True, p=probs)
        unique, counts = np.unique(sampled, return_counts=True)
        return unique, counts.astype(np.float64) / k

    def _expand_from_priors(self, node: MCTSNode,
                            actions_np: np.ndarray, priors_np: np.ndarray):
        """Populate parallel-array storage from pre-computed numpy priors.

        Pure numpy — no GPU ops, no CPU↔GPU transfer. Callers batch the softmax
        and .cpu() transfer across all games in a sim step, then hand off here.

        ``priors_np`` is the PUCT prior over the children being installed.
        Under Sampled MuZero (Hubert 2021 Proposed Modification) sampled call
        sites pass β̂(a) = count(a)/K from i.i.d. multinomial sampling; under
        full expansion it is the full masked softmax. Root Dirichlet noise is
        applied later via ``_add_dirichlet_noise`` on ``child_priors``.

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
            tv = node.child_terminal_value
            if tv is not None and tv[idx] == tv[idx]:  # not NaN → forced-draw terminal child
                child.terminal_value = float(tv[idx])
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
        term = node.child_terminal_value.tolist() if node.child_terminal_value is not None else None

        best_score = -float("inf")
        best_idx = -1
        for i in range(len(priors)):
            v = visits[i]
            prior_score = pb_c * priors[i] * sqrt_n / (1.0 + v)
            if term is not None and term[i] == term[i]:  # forced-draw terminal: fixed Q, all visit counts
                raw_q = term[i]
                value_score = (raw_q - mm_min) / mm_span if has_range else raw_q
            elif v > 0:
                # Mover-POV Q at this child:
                #   Q(parent POV) = child.reward + γ * V(child)_parent_POV
                #                 = child.reward + γ * (-child.value_child_POV)
                #                 = child.reward - γ * child.value
                # child.value_sums/v is stored in the child's own POV (see
                # _backpropagate) — flip via the -γ · child.value term.
                raw_q = rewards[i] - discount * (value_sums[i] / v)
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
        # Mover-POV Q (see _select_child_loop for the derivation).
        raw_q = rewards - self.config.discount * child_values

        if min_max_stats.maximum > min_max_stats.minimum:
            normalized_q = (raw_q - min_max_stats.minimum) / (min_max_stats.maximum - min_max_stats.minimum)
        else:
            normalized_q = raw_q
        value_scores = np.where(visits > 0, normalized_q, 0.0)

        # Root-terminal-draws: forced-draw children use the fixed draw value as Q at
        # ALL visit counts (overrides the visits>0 gate), so a winning side avoids
        # the repeating move and a losing side keeps it.
        term = node.child_terminal_value
        if term is not None:
            tmask = ~np.isnan(term)
            if tmask.any():
                if min_max_stats.maximum > min_max_stats.minimum:
                    tnorm = (term - min_max_stats.minimum) / (min_max_stats.maximum - min_max_stats.minimum)
                else:
                    tnorm = term
                value_scores = np.where(tmask, tnorm, value_scores)

        scores = prior_scores + value_scores
        best_idx = int(np.argmax(scores))
        return int(node.child_actions[best_idx]), best_idx, self._get_or_create_child(node, best_idx)

    def _backpropagate(self, search_path: list[MCTSNode], path_indices: list[int],
                       value: float, min_max_stats: MinMaxStats):
        """Backprop the leaf value up the search path and sync parent arrays.

        path_indices[i] is the index of search_path[i] in search_path[i-1]'s
        child arrays (or -1 for the root, which has no parent).

        Sign convention (mover-POV, matches game.step + the trainer reward
        target). On entry ``value`` is the leaf network's V_leaf in the leaf
        player's POV. At each iteration ``value`` is in the frame of ``node``,
        so it is added to ``node.value_sum`` directly. To re-express it in the
        parent's frame for the next iteration we apply
        ``value = node.reward - γ · value`` — node.reward is from the parent
        (mover) POV already, and -γ·value flips the leaf's-POV cumulative into
        the parent's POV (two-player zero-sum: each ply up swaps perspectives).
        """
        n = len(search_path)
        for depth in range(n - 1, -1, -1):
            node = search_path[depth]
            node.visit_count += 1
            node.value_sum += value

            if depth > 0:
                parent = search_path[depth - 1]
                idx = path_indices[depth]
                parent.child_visits[idx] = node.visit_count
                parent.child_value_sums[idx] = node.value_sum
                parent.child_rewards[idx] = node.reward

            if node.visit_count > 0:
                # Mover-POV Q used by selection — must match _select_child.
                min_max_stats.update(node.reward - self.config.discount * node.value)

            value = node.reward - self.config.discount * value


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

    If ``config.sample_k`` is set, implements Hubert 2021 §5.1 Proposed
    Modification: at each node with ``sample_k`` < legal-action-space, sample
    K actions i.i.d. *with replacement* from β = π and use β̂(a) = count(a)/K
    as the PUCT prior over the *unique* sampled actions. Training target is
    raw ``N(a)/ΣN`` over sampled children — Theorem 1 gives that this
    approximates the improved policy Î_β π. No post-hoc β̂/β correction at
    target time. ``sample_k=None`` expands the full action space at each leaf
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
        # Mover-POV Q (matches _select_child).
        raw_q = rewards - self.config.discount * child_values
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
        forced_draw_actions: list | None = None,
    ) -> list[MCTSNode]:
        """forced_draw_actions: optional per-game container (set/list) of root action
        indices that lead to a draw-by-repetition / 50-move. Those root children are
        treated as terminal with the contempt draw value (root-terminal-draws override):
        a winning side plays on, a losing side keeps the draw. Caller derives them from
        the real board (the agent's own move history — not a simulator oracle)."""
        n = len(observations)
        roots = [MCTSNode() for _ in range(n)]
        min_max_stats = [MinMaxStats() for _ in range(n)]
        draw_score = float(getattr(self.config, "draw_score", 0.0))

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
                # Hubert 2021 §5.1 Proposed Modification: sample K i.i.d. with
                # replacement from β = π over legal actions. Use β̂(a) = count(a)/K
                # as the PUCT prior over the unique sampled actions. K_unique ≤ K.
                unique_local, beta_hat = self._sample_iid_with_replacement(
                    full_priors, sample_k
                )
                actions_np = legals_np[unique_local]
                priors_np = beta_hat
            else:
                actions_np = legals_np
                priors_np = full_priors
            self._expand_from_priors(roots[g], actions_np, priors_np)
            if add_noise and roots[g].children:
                self._add_dirichlet_noise(roots[g])
            if forced_draw_actions is not None and forced_draw_actions[g]:
                fd = forced_draw_actions[g]
                ctv = np.full(len(roots[g].child_actions), np.nan, dtype=np.float64)
                for ci, a in enumerate(roots[g].child_actions):
                    if int(a) in fd:
                        ctv[ci] = draw_score
                roots[g].child_terminal_value = ctv

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
                # Hubert 2021 §5.1 Proposed Modification: sample K i.i.d. with
                # replacement from β = π. Aggregate per-game to unique actions
                # with β̂(a) = count(a)/K as the PUCT prior. K_unique ≤ K varies
                # per game.
                sampled_idx_gpu = torch.multinomial(
                    probs_gpu, sample_k, replacement=True
                )
                sampled_idx_np = sampled_idx_gpu.detach().cpu().numpy().astype(np.int64)
                topk_actions_list: list[np.ndarray] = []
                topk_priors_list: list[np.ndarray] = []
                for sample_row in sampled_idx_np:
                    unique, counts = np.unique(sample_row, return_counts=True)
                    topk_actions_list.append(unique)
                    topk_priors_list.append(
                        counts.astype(np.float64) / sample_k
                    )
            else:
                probs_np = probs_gpu.detach().cpu().numpy().astype(np.float64)

            for g in range(n):
                leaf = search_paths[g][-1]
                if leaf.terminal_value is not None:
                    # Forced-draw terminal child: do NOT expand. Back up the draw
                    # value (0.0 — a draw ≈ 0 for the value target; the contempt
                    # draw_score acts only in selection, matching draw_score's
                    # "tree-side only" semantics). The recurrent_inference for this
                    # game this step is discarded (wasteful but rare; correctness > speed).
                    self._backpropagate(search_paths[g], path_indices_all[g],
                                        0.0, min_max_stats[g])
                    continue
                leaf.hidden_state = next_hiddens[g]
                leaf.reward = rewards_list[g]
                if use_sampled_leaf:
                    self._expand_from_priors(
                        leaf, topk_actions_list[g], topk_priors_list[g]
                    )
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
    Modification) the search already accounts for the sampling via β̂(a) =
    count(a)/K as the PUCT prior, so no post-hoc β̂/β correction is applied
    here — Theorem 1 gives that raw visit counts already approximate the
    improved policy Î_β π.

    Returns:
        action: selected action index
        action_probs: dense policy distribution over [0, max(actions)+1)
    """
    actions = root.child_actions
    visits = root.child_visits.astype(np.float64)

    # bug_hunt_2026_06_13.md §B: the dense TRAINING TARGET is the raw visit
    # fraction N(a)/ΣN(a), temperature-independent for ALL T (matches the
    # docstring above and reanalyze's T=1.0 convention). Temperature affects
    # only WHICH action we sample/play, never the stored target.
    raw_total = visits.sum()
    if raw_total <= 0:
        raw_probs = np.ones(len(actions)) / len(actions)
    else:
        raw_probs = visits / raw_total

    if temperature == 0:
        action = int(actions[int(np.argmax(visits))])
    else:
        visits_temp = visits ** (1.0 / temperature)
        total_v = visits_temp.sum()
        if total_v <= 0:
            select_probs = np.ones(len(actions)) / len(actions)
        else:
            select_probs = visits_temp / total_v
        action = int(np.random.choice(actions, p=select_probs))

    max_action = int(actions.max()) + 1
    action_probs = np.zeros(max_action, dtype=np.float32)
    action_probs[actions] = raw_probs
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

    # Mover-POV Q at each sampled child (matches PUCT).
    with np.errstate(invalid="ignore", divide="ignore"):
        sampled_child_values = np.where(
            sampled_visits > 0,
            sampled_value_sums / np.maximum(sampled_visits, 1.0),
            0.0,
        )
    raw_q_sampled = sampled_rewards - config.discount * sampled_child_values

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
