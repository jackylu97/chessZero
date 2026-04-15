"""Monte Carlo Tree Search for MuZero."""

import math
from dataclasses import dataclass, field

import numpy as np
import torch


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

        # Backpropagate initial value
        root.visit_count += 1
        root.value_sum += value.item()

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            current_player_sign = 1

            # SELECT: traverse tree until we find an unexpanded node
            while node.expanded():
                action, child = self._select_child(node)
                node = child
                search_path.append(node)
                current_player_sign *= -1  # alternating players

            # EXPAND and EVALUATE
            if node.hidden_state is not None:
                # This shouldn't happen for leaf nodes; handle it gracefully
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

                leaf_value = value.item()
            else:
                # First visit to this node - need to run dynamics
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

                leaf_value = value.item()

            # BACKPROPAGATE
            self._backpropagate(search_path, leaf_value, current_player_sign)

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

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Select child with highest UCB score (PUCT formula)."""
        best_score = -float("inf")
        best_action = -1
        best_child = None

        total_visits = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(total_visits + 1)

        for action, child in node.children.items():
            # PUCT formula
            prior_score = self.config.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            value_score = -child.value if child.visit_count > 0 else 0  # negate for opponent
            score = value_score + prior_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backpropagate(self, search_path: list[MCTSNode], value: float, current_player_sign: int):
        """Backpropagate value through the search path."""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value * current_player_sign
            current_player_sign *= -1
            value = node.reward + self.config.discount * value


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
