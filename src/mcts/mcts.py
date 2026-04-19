"""Monte Carlo Tree Search for MuZero."""

import math
from dataclasses import dataclass, field

import numpy as np
import torch


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


@dataclass
class MCTSNode:
    """A node in the MCTS search tree.

    State is double-booked for performance:
      - Each child has its own `prior`, `visit_count`, `value_sum`, `reward`
        attributes (used by the Python-loop selection path at small K).
      - The parent holds parallel numpy arrays `child_priors`, `child_visits`,
        `child_value_sums`, `child_rewards` (used by the vectorized selection
        path at large K).
    `_backpropagate` keeps both copies in sync after each simulation.
    """
    hidden_state: torch.Tensor | None = None
    prior: float = 0.0
    reward: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: list["MCTSNode"] = field(default_factory=list)
    child_actions: np.ndarray | None = None       # (K,) int64
    child_priors: np.ndarray | None = None        # (K,) float64
    child_visits: np.ndarray | None = None        # (K,) float64 (stored float for vectorized math)
    child_rewards: np.ndarray | None = None       # (K,) float64
    child_value_sums: np.ndarray | None = None    # (K,) float64

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
            self._expand(node, policy_logits.squeeze(0), all_actions,
                         top_k=getattr(self.config, "leaf_top_k", None))

            self._backpropagate(search_path, path_indices, value.item(), min_max_stats)

        return root

    def _add_dirichlet_noise(self, node: MCTSNode):
        """Mix Dirichlet noise into a node's child priors (applied only at root)."""
        k = len(node.children)
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * k)
        eps = self.config.dirichlet_epsilon
        node.child_priors = node.child_priors * (1.0 - eps) + noise * eps
        # Mirror into child.prior so the loop selection path sees the noise too.
        for i, child in enumerate(node.children):
            child.prior = float(node.child_priors[i])

    def _expand(self, node: MCTSNode, policy_logits: torch.Tensor,
                legal_actions: list[int], top_k: int | None = None):
        """Expand node with children, populating the parallel-array storage.

        Does its own GPU softmax + CPU transfer. `BatchedMCTS.run_batch` bypasses
        this for per-sim leaves — it batches softmax/topk across all N games into
        one transfer, then calls `_expand_from_priors` below.
        """
        policy = torch.full_like(policy_logits, float("-inf"))
        legal_idx = torch.tensor(legal_actions, dtype=torch.long)
        policy[legal_idx] = policy_logits[legal_idx]
        policy = torch.softmax(policy, dim=0)

        if top_k is not None and len(legal_actions) > top_k:
            topk = torch.topk(policy, top_k)
            actions_np = topk.indices.detach().cpu().numpy().astype(np.int64)
            priors_np = topk.values.detach().cpu().numpy().astype(np.float64)
        else:
            actions_np = np.asarray(legal_actions, dtype=np.int64)
            priors_np = policy[torch.from_numpy(actions_np)].detach().cpu().numpy().astype(np.float64)

        self._expand_from_priors(node, actions_np, priors_np)

    def _expand_from_priors(self, node: MCTSNode,
                            actions_np: np.ndarray, priors_np: np.ndarray):
        """Populate parallel-array storage from pre-computed numpy priors.

        Pure numpy — no GPU ops, no CPU↔GPU transfer. Callers batch the softmax
        and .cpu() transfer across all games in a sim step, then hand off here.
        """
        k = len(actions_np)
        node.child_actions = actions_np
        node.child_priors = priors_np.astype(np.float64, copy=False)
        node.child_visits = np.zeros(k, dtype=np.float64)
        node.child_rewards = np.zeros(k, dtype=np.float64)
        node.child_value_sums = np.zeros(k, dtype=np.float64)
        node.children = [MCTSNode(prior=float(priors_np[i])) for i in range(k)]

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

        Reads child stats from the child object attributes (attribute access is ~2x
        faster than numpy scalar indexing for this size of loop).
        """
        pb_c = math.log((node.visit_count + self.PB_C_BASE + 1) / self.PB_C_BASE) + self.PB_C_INIT
        sqrt_n = math.sqrt(node.visit_count)
        discount = self.config.discount
        mm_min, mm_max = min_max_stats.minimum, min_max_stats.maximum
        has_range = mm_max > mm_min
        mm_span = mm_max - mm_min if has_range else 1.0

        best_score = -float("inf")
        best_idx = -1
        for i, child in enumerate(node.children):
            v = child.visit_count
            prior_score = pb_c * child.prior * sqrt_n / (1.0 + v)
            if v > 0:
                raw_q = -child.reward - discount * (child.value_sum / v)
                value_score = (raw_q - mm_min) / mm_span if has_range else raw_q
            else:
                value_score = 0.0
            score = prior_score + value_score
            if score > best_score:
                best_score = score
                best_idx = i
        return int(node.child_actions[best_idx]), best_idx, node.children[best_idx]

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
        return int(node.child_actions[best_idx]), best_idx, node.children[best_idx]

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


class BatchedMCTS(MCTS):
    """MCTS over N games simultaneously, batching all network calls.

    Sync budget per simulation: **3 GPU→CPU transfers** regardless of N —
    one bulk `.tolist()` for rewards, one for values, one `.cpu()` for
    policy top-K (or full softmax if no `leaf_top_k`). The previous
    implementation did 3N transfers (one per game per step); at N=64 this
    is a ~60× reduction in sync-barrier count.
    """

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

        obs_batch = torch.stack(observations).to(self.device)
        hidden_batch_gpu, policy_batch, _ = self.network.initial_inference(obs_batch)

        # One bulk transfer of all root logits; mask+softmax per game in numpy.
        root_logits_cpu = policy_batch.detach().cpu().numpy().astype(np.float64)
        for g in range(n):
            roots[g].hidden_state = hidden_batch_gpu[g]
            legals_np = np.asarray(legal_actions_list[g], dtype=np.int64)
            priors_np = _masked_softmax_np(root_logits_cpu[g], legals_np)
            self._expand_from_priors(roots[g], legals_np, priors_np)
            if add_noise and roots[g].children:
                self._add_dirichlet_noise(roots[g])

        action_space_size = self.game.action_space_size
        leaf_top_k = getattr(self.config, "leaf_top_k", None)
        use_topk = leaf_top_k is not None and leaf_top_k < action_space_size
        all_actions_np = np.arange(action_space_size, dtype=np.int64)

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
            if use_topk:
                topk = torch.topk(probs_gpu, leaf_top_k, dim=-1)
                topk_actions_np = topk.indices.detach().cpu().numpy().astype(np.int64)
                topk_priors_np = topk.values.detach().cpu().numpy().astype(np.float64)
            else:
                probs_np = probs_gpu.detach().cpu().numpy().astype(np.float64)

            for g in range(n):
                leaf = search_paths[g][-1]
                leaf.hidden_state = next_hiddens[g]
                leaf.reward = rewards_list[g]
                if use_topk:
                    self._expand_from_priors(leaf, topk_actions_np[g], topk_priors_np[g])
                else:
                    self._expand_from_priors(leaf, all_actions_np, probs_np[g])
                self._backpropagate(search_paths[g], path_indices_all[g],
                                    values_list[g], min_max_stats[g])

        return roots


def select_action(root: MCTSNode, temperature: float = 1.0) -> tuple[int, list[float]]:
    """Select action from MCTS visit counts using temperature.

    Returns:
        action: selected action index
        action_probs: probability distribution over all children (dense up to max action)
    """
    actions = root.child_actions
    visits = root.child_visits.astype(np.float64)

    if temperature == 0:
        best = int(np.argmax(visits))
        probs = np.zeros(len(actions))
        probs[best] = 1.0
        action = int(actions[best])
    else:
        visits_temp = visits ** (1.0 / temperature)
        total = visits_temp.sum()
        if total <= 0:
            probs = np.ones(len(actions)) / len(actions)
        else:
            probs = visits_temp / total
        action = int(np.random.choice(actions, p=probs))

    max_action = int(actions.max()) + 1
    action_probs = np.zeros(max_action, dtype=np.float64)
    action_probs[actions] = probs
    return action, action_probs.tolist()
