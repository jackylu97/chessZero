"""Latent state visualization and analysis tools (Phase 3)."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class LatentAnalyzer:
    """Extract and analyze hidden state representations across games."""

    def __init__(self, network, device: str = "cpu"):
        self.network = network
        self.device = device
        self.network.eval()

    @torch.no_grad()
    def extract_hidden_states(self, observations: list[torch.Tensor],
                               game_name: str | None = None) -> np.ndarray:
        """Extract hidden states for a batch of observations.

        Args:
            observations: list of (C, H, W) tensors
            game_name: required for MultiGameMuZeroNetwork

        Returns:
            hidden_states: (N, hidden_dim) flattened hidden representations
        """
        obs_batch = torch.stack(observations).to(self.device)
        if game_name is not None:
            hidden, _, _ = self.network.initial_inference(obs_batch, game_name)
        else:
            hidden, _, _ = self.network.initial_inference(obs_batch)

        # Flatten spatial dimensions
        return hidden.view(hidden.shape[0], -1).cpu().numpy()

    def compute_tsne(self, hidden_states: np.ndarray, perplexity: float = 30.0,
                     n_components: int = 2) -> np.ndarray:
        """Reduce hidden states to 2D/3D using t-SNE."""
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        return tsne.fit_transform(hidden_states)

    def compute_umap(self, hidden_states: np.ndarray, n_neighbors: int = 15,
                     n_components: int = 2) -> np.ndarray:
        """Reduce hidden states to 2D/3D using UMAP."""
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=42)
        return reducer.fit_transform(hidden_states)

    def train_probe(self, hidden_states: np.ndarray, labels: np.ndarray,
                    probe_type: str = "linear") -> dict:
        """Train a linear probe on hidden states to predict game properties.

        Args:
            hidden_states: (N, hidden_dim)
            labels: (N,) classification labels or (N,) regression targets
            probe_type: "linear" for logistic regression, "regression" for linear regression

        Returns:
            dict with 'accuracy' or 'r2' and 'model'
        """
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import cross_val_score

        if probe_type == "linear":
            model = LogisticRegression(max_iter=1000, random_state=42)
            scores = cross_val_score(model, hidden_states, labels, cv=5, scoring="accuracy")
            model.fit(hidden_states, labels)
            return {"accuracy": scores.mean(), "std": scores.std(), "model": model}
        else:
            model = Ridge(alpha=1.0)
            scores = cross_val_score(model, hidden_states, labels, cv=5, scoring="r2")
            model.fit(hidden_states, labels)
            return {"r2": scores.mean(), "std": scores.std(), "model": model}

    def cross_game_similarity(self, states_a: np.ndarray, states_b: np.ndarray) -> dict:
        """Compute representational similarity between two sets of hidden states.

        Uses Centered Kernel Alignment (CKA) to compare representations.
        """
        # Linear CKA
        def _centering(K):
            n = K.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            return H @ K @ H

        K_a = states_a @ states_a.T
        K_b = states_b @ states_b.T

        K_a_c = _centering(K_a)
        K_b_c = _centering(K_b)

        hsic = np.sum(K_a_c * K_b_c)
        norm_a = np.sqrt(np.sum(K_a_c * K_a_c))
        norm_b = np.sqrt(np.sum(K_b_c * K_b_c))

        cka = hsic / (norm_a * norm_b + 1e-10)
        return {"cka": float(cka)}

    def visualize_to_tensorboard(self, hidden_states: np.ndarray,
                                  labels: list[str], metadata: list[str],
                                  log_dir: str = "runs/latent_viz"):
        """Log hidden state embeddings to TensorBoard for interactive exploration."""
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_embedding(
            torch.from_numpy(hidden_states),
            metadata=metadata,
            tag="hidden_states",
        )
        writer.close()


def collect_game_states(game, num_states: int = 500, max_moves: int = 200) -> list[dict]:
    """Play random games and collect diverse board positions.

    Returns list of dicts with 'observation', 'game_name', 'move_number',
    'game_phase' (opening/mid/end), etc.
    """
    import random
    states = []

    while len(states) < num_states:
        state = game.reset()
        move_count = 0

        while not state.done and move_count < max_moves:
            obs = game.to_tensor(state)
            legal = game.legal_actions(state)
            total_moves = max_moves  # approximate game length

            # Classify game phase
            progress = move_count / max(total_moves, 1)
            if progress < 0.2:
                phase = "opening"
            elif progress < 0.7:
                phase = "midgame"
            else:
                phase = "endgame"

            states.append({
                "observation": obs,
                "move_number": move_count,
                "game_phase": phase,
                "current_player": state.current_player,
            })

            if len(states) >= num_states:
                break

            action = random.choice(legal)
            state, _, _ = game.step(state, action)
            move_count += 1

    return states[:num_states]
