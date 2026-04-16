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
    """A node in the MCTS search tree."""
    hidden_state: torch.Tensor | None = None  # (C, H, W)
    reward: float = 0.0
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """PUCT-based MCTS for MuZero."""

    # Constants from the MuZero/AlphaZero paper
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
        """Run MCTS simulations from root observation.

        Args:
            observation: (C, H, W) tensor
            legal_actions: list of legal action indices
            add_noise: whether to add Dirichlet noise at root

        Returns:
            Root MCTSNode with search statistics.
        """
        root = MCTSNode()
        min_max_stats = MinMaxStats()

        # Initial inference
        obs_batch = observation.unsqueeze(0).to(self.device)
        hidden, policy_logits, value = self.network.initial_inference(obs_batch)

        root.hidden_state = hidden.squeeze(0)

        # Expand root
        self._expand(root, policy_logits.squeeze(0), legal_actions)

        # Add Dirichlet noise at root for exploration
        if add_noise and len(root.children) > 0:
            noise = np.random.dirichlet(
                [self.config.dirichlet_alpha] * len(root.children)
            )
            eps = self.config.dirichlet_epsilon
            for i, (action, child) in enumerate(root.children.items()):
                child.prior = child.prior * (1 - eps) + noise[i] * eps

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            # SELECT: traverse tree until we find an unexpanded node
            while node.expanded():
                action, child = self._select_child(node, min_max_stats)
                node = child
                search_path.append(node)

            # EXPAND and EVALUATE
            parent = search_path[-2]
            parent_hidden = parent.hidden_state.unsqueeze(0).to(self.device)
            action_tensor = torch.tensor([action], device=self.device)
            next_hidden, reward, policy_logits, value = self.network.recurrent_inference(
                parent_hidden, action_tensor
            )
            node.hidden_state = next_hidden.squeeze(0)
            node.reward = reward.item()

            # For leaf expansion, we use all actions as potentially legal
            # (MCTS in latent space doesn't have access to game rules at leaves)
            all_actions = list(range(self.game.action_space_size))
            self._expand(node, policy_logits.squeeze(0), all_actions)

            # BACKPROPAGATE
            self._backpropagate(search_path, value.item(), min_max_stats)

        return root

    def _expand(self, node: MCTSNode, policy_logits: torch.Tensor, legal_actions: list[int]):
        """Expand node with children for each legal action."""
        # Mask illegal actions and compute policy
        policy = torch.full_like(policy_logits, float("-inf"))
        legal_idx = torch.tensor(legal_actions, dtype=torch.long)
        policy[legal_idx] = policy_logits[legal_idx]
        policy = torch.softmax(policy, dim=0)

        for action in legal_actions:
            node.children[action] = MCTSNode(prior=policy[action].item())

    def _select_child(self, node: MCTSNode, min_max_stats: MinMaxStats) -> tuple[int, MCTSNode]:
        """Select child with highest UCB score (PUCT formula from AlphaZero/MuZero)."""
        best_score = -float("inf")
        best_action = -1
        best_child = None

        # Dynamic exploration constant (grows logarithmically with visits)
        pb_c = math.log((node.visit_count + self.PB_C_BASE + 1) / self.PB_C_BASE) + self.PB_C_INIT

        for action, child in node.children.items():
            # Prior score (exploration)
            prior_score = pb_c * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)

            # Value score (exploitation) — includes reward from transition
            if child.visit_count > 0:
                # Q-value = immediate reward + discounted future value, negated for opponent
                raw_value = -child.reward - self.config.discount * child.value
                value_score = min_max_stats.normalize(raw_value)
            else:
                value_score = 0.0

            score = prior_score + value_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backpropagate(self, search_path: list[MCTSNode], value: float, min_max_stats: MinMaxStats):
        """Backpropagate value through the search path.

        Value alternates sign for two-player games (opponent's gain = our loss).
        """
        for i, node in enumerate(reversed(search_path)):
            # Alternate perspective: even depth = current player, odd = opponent
            sign = 1.0 if (i % 2 == 0) else -1.0
            node.visit_count += 1
            node.value_sum += value * sign

            # Update MinMaxStats with the Q-value that UCB will use for this node
            # (from the parent's perspective: -reward - discount * value)
            if node.visit_count > 0:
                min_max_stats.update(-node.reward - self.config.discount * node.value)

            # Bootstrap value through the tree
            value = node.reward + self.config.discount * value


class BatchedMCTS(MCTS):
    """MCTS over N games simultaneously, batching all network calls across games.

    Each simulation step does one recurrent_inference call of batch size N
    instead of N calls of batch size 1. For small models on GPU this gives a
    large throughput improvement since kernel launch overhead dominates over
    actual compute at batch size 1.
    """

    @torch.no_grad()
    def run_batch(
        self,
        observations: list[torch.Tensor],
        legal_actions_list: list[list[int]],
        add_noise: bool = True,
    ) -> list[MCTSNode]:
        """Run MCTS for N games in parallel.

        Args:
            observations: list of N (C, H, W) tensors
            legal_actions_list: list of N legal-action lists
            add_noise: whether to add Dirichlet noise at roots

        Returns:
            list of N root MCTSNodes with search statistics
        """
        n = len(observations)
        roots = [MCTSNode() for _ in range(n)]
        min_max_stats = [MinMaxStats() for _ in range(n)]

        # Batch initial inference for all N roots
        obs_batch = torch.stack(observations).to(self.device)  # (N, C, H, W)
        hidden_batch, policy_batch, value_batch = self.network.initial_inference(obs_batch)

        for g in range(n):
            roots[g].hidden_state = hidden_batch[g]
            self._expand(roots[g], policy_batch[g], legal_actions_list[g])
            if add_noise and roots[g].children:
                noise = np.random.dirichlet(
                    [self.config.dirichlet_alpha] * len(roots[g].children)
                )
                eps = self.config.dirichlet_epsilon
                for i, child in enumerate(roots[g].children.values()):
                    child.prior = child.prior * (1 - eps) + noise[i] * eps

        all_actions = list(range(self.game.action_space_size))

        for _ in range(self.config.num_simulations):
            search_paths = []
            parent_hiddens = []
            leaf_actions = []

            # SELECT phase — pure CPU, one pass per game
            for g in range(n):
                node = roots[g]
                path = [node]
                last_action = None
                while node.expanded():
                    last_action, child = self._select_child(node, min_max_stats[g])
                    node = child
                    path.append(node)
                search_paths.append(path)
                parent_hiddens.append(path[-2].hidden_state)
                leaf_actions.append(last_action)

            # EXPAND — single batched network call
            hidden_batch = torch.stack(parent_hiddens).to(self.device)       # (N, C, H, W)
            action_batch = torch.tensor(leaf_actions, device=self.device)    # (N,)
            next_hiddens, rewards, policy_batch, value_batch = \
                self.network.recurrent_inference(hidden_batch, action_batch)

            # BACKPROPAGATE — distribute results
            for g in range(n):
                leaf = search_paths[g][-1]
                leaf.hidden_state = next_hiddens[g]
                leaf.reward = rewards[g].item()
                self._expand(leaf, policy_batch[g], all_actions)
                self._backpropagate(search_paths[g], value_batch[g].item(), min_max_stats[g])

        return roots


def select_action(root: MCTSNode, temperature: float = 1.0) -> tuple[int, list[float]]:
    """Select action from MCTS visit counts using temperature.

    Returns:
        action: selected action index
        action_probs: probability distribution over all children
    """
    actions = list(root.children.keys())
    visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)

    if temperature == 0:
        # Greedy
        action = actions[np.argmax(visits)]
        probs = np.zeros(len(actions))
        probs[np.argmax(visits)] = 1.0
    else:
        # Temperature-scaled distribution
        visits_temp = visits ** (1.0 / temperature)
        probs = visits_temp / visits_temp.sum()
        action = np.random.choice(actions, p=probs)

    # Build full action probability vector
    max_action = max(actions) + 1
    action_probs = np.zeros(max_action)
    for a, p in zip(actions, probs):
        action_probs[a] = p

    return action, action_probs.tolist()
