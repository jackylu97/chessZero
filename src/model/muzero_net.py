"""MuZero neural networks: representation, dynamics, prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ResidualBlock, mlp_head, support_to_scalar, inverse_scalar_transform


class RepresentationNetwork(nn.Module):
    """h(observation) -> hidden_state

    Maps game-specific observation to a latent representation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_planes: int,
        num_blocks: int,
        latent_h: int,
        latent_w: int,
        input_h: int,
        input_w: int,
    ):
        super().__init__()
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.needs_resize = (input_h != latent_h) or (input_w != latent_w)

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_planes),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_planes) for _ in range(num_blocks)]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.projection(obs)
        if self.needs_resize:
            x = F.interpolate(x, size=(self.latent_h, self.latent_w), mode="bilinear", align_corners=False)
        x = self.blocks(x)
        x = _min_max_normalize(x)
        return x


class DynamicsNetwork(nn.Module):
    """g(hidden_state, action) -> (next_hidden_state, reward)

    Action is encoded as a normalized scalar plane broadcast over latent dims.
    Includes a residual skip connection from input hidden state.
    """

    def __init__(
        self,
        hidden_planes: int,
        num_blocks: int,
        action_space_size: int,
        latent_h: int,
        latent_w: int,
        fc_hidden: int,
        reward_support_size: int = 1,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.reward_support_size = reward_support_size

        # Conv to reduce channels after concatenating action plane
        self.conv_in = nn.Conv2d(hidden_planes + 1, hidden_planes, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(hidden_planes)

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_planes) for _ in range(num_blocks)]
        )

        # Reward head outputs categorical distribution
        reward_out = 2 * reward_support_size + 1
        self.reward_head = nn.Sequential(
            nn.Conv2d(hidden_planes, 1, 1),
            nn.Flatten(),
            mlp_head(latent_h * latent_w, fc_hidden, reward_out),
        )

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: (B, C, H, W)
            action: (B,) integer action indices
        Returns:
            next_hidden_state: (B, C, H, W)
            reward_logits: (B, 2*support_size+1)
        """
        b = hidden_state.shape[0]
        # Encode action as a normalized scalar broadcast over spatial dims
        action_plane = action.float() / self.action_space_size
        action_plane = action_plane.view(b, 1, 1, 1).expand(b, 1, self.latent_h, self.latent_w)

        x = torch.cat([hidden_state, action_plane], dim=1)
        x = self.conv_in(x)
        x = self.bn_in(x)
        # Residual skip connection from input hidden state
        x = F.relu(x + hidden_state)
        x = self.blocks(x)
        x = _min_max_normalize(x)

        reward_logits = self.reward_head(x)
        return x, reward_logits


class PredictionNetwork(nn.Module):
    """f(hidden_state) -> (policy, value)"""

    def __init__(
        self,
        hidden_planes: int,
        action_space_size: int,
        latent_h: int,
        latent_w: int,
        fc_hidden: int,
        value_support_size: int = 1,
    ):
        super().__init__()
        self.value_support_size = value_support_size

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_planes, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * latent_h * latent_w, action_space_size),
        )

        # Value head outputs categorical distribution
        value_out = 2 * value_support_size + 1
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_planes, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            mlp_head(latent_h * latent_w, fc_hidden, value_out),
        )

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            policy_logits: (B, action_space_size)
            value_logits: (B, 2*support_size+1)
        """
        return self.policy_head(hidden_state), self.value_head(hidden_state)


class MuZeroNetwork(nn.Module):
    """Full MuZero network combining representation, dynamics, prediction.

    Value and reward are represented as categorical distributions over a
    discrete support {-support_size, ..., 0, ..., support_size}, following
    the MuZero paper. This is more stable than scalar regression.
    """

    def __init__(
        self,
        observation_channels: int,
        action_space_size: int,
        hidden_planes: int = 64,
        num_blocks: int = 4,
        latent_h: int = 6,
        latent_w: int = 7,
        input_h: int = 6,
        input_w: int = 7,
        fc_hidden: int = 128,
        value_support_size: int = 10,
        reward_support_size: int = 1,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.value_support_size = value_support_size
        self.reward_support_size = reward_support_size

        self.representation = RepresentationNetwork(
            observation_channels, hidden_planes, num_blocks,
            latent_h, latent_w, input_h, input_w,
        )
        self.dynamics = DynamicsNetwork(
            hidden_planes, num_blocks, action_space_size,
            latent_h, latent_w, fc_hidden, reward_support_size,
        )
        self.prediction = PredictionNetwork(
            hidden_planes, action_space_size,
            latent_h, latent_w, fc_hidden, value_support_size,
        )

    def initial_inference(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run representation + prediction on observation.

        Returns: (hidden_state, policy_logits, value_scalar)
        """
        hidden = self.representation(obs)
        policy_logits, value_logits = self.prediction(hidden)
        # Decode categorical value to scalar for MCTS
        value = self._value_logits_to_scalar(value_logits)
        return hidden, policy_logits, value

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run dynamics + prediction on hidden state + action.

        Returns: (next_hidden_state, reward_scalar, policy_logits, value_scalar)
        """
        next_hidden, reward_logits = self.dynamics(hidden_state, action)
        policy_logits, value_logits = self.prediction(next_hidden)
        # Decode to scalars for MCTS
        value = self._value_logits_to_scalar(value_logits)
        reward = self._reward_logits_to_scalar(reward_logits)
        return next_hidden, reward, policy_logits, value

    def initial_inference_logits(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like initial_inference but returns raw logits (for training).

        Returns: (hidden_state, policy_logits, value_logits)
        """
        hidden = self.representation(obs)
        policy_logits, value_logits = self.prediction(hidden)
        return hidden, policy_logits, value_logits

    def recurrent_inference_logits(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like recurrent_inference but returns raw logits (for training).

        Returns: (next_hidden_state, reward_logits, policy_logits, value_logits)
        """
        next_hidden, reward_logits = self.dynamics(hidden_state, action)
        policy_logits, value_logits = self.prediction(next_hidden)
        return next_hidden, reward_logits, policy_logits, value_logits

    def _value_logits_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        scalar = support_to_scalar(logits, self.value_support_size)
        return inverse_scalar_transform(scalar).unsqueeze(-1)

    def _reward_logits_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        scalar = support_to_scalar(logits, self.reward_support_size)
        return inverse_scalar_transform(scalar).unsqueeze(-1)


class MultiGameMuZeroNetwork(nn.Module):
    """Multi-game MuZero with shared backbone and game-specific heads.

    Phase 2: Shared residual backbone processes all games.
    Per-game input projections and output heads handle different
    observation/action spaces.
    """

    def __init__(self, game_configs: dict, hidden_planes: int = 128,
                 num_blocks: int = 8, latent_h: int = 8, latent_w: int = 8,
                 fc_hidden: int = 128, game_id_dim: int = 16,
                 value_support_size: int = 10, reward_support_size: int = 1):
        """
        Args:
            game_configs: dict mapping game_name -> {
                'obs_channels': int, 'action_space': int,
                'input_h': int, 'input_w': int
            }
        """
        super().__init__()
        self.game_names = list(game_configs.keys())
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.hidden_planes = hidden_planes
        self.value_support_size = value_support_size
        self.reward_support_size = reward_support_size

        value_out = 2 * value_support_size + 1
        reward_out = 2 * reward_support_size + 1

        # Game ID embeddings
        self.game_embeddings = nn.Embedding(len(self.game_names), game_id_dim)

        # Per-game input projections
        self.input_projections = nn.ModuleDict()
        for name, cfg in game_configs.items():
            self.input_projections[name] = nn.Sequential(
                nn.Conv2d(cfg["obs_channels"], hidden_planes, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_planes),
                nn.ReLU(),
            )

        # Shared backbone (representation)
        self.shared_repr_blocks = nn.Sequential(
            *[ResidualBlock(hidden_planes) for _ in range(num_blocks)]
        )

        # Shared dynamics backbone
        self.dynamics_conv_in = nn.Conv2d(hidden_planes + 1, hidden_planes, 3, padding=1, bias=False)
        self.dynamics_bn_in = nn.BatchNorm2d(hidden_planes)
        self.shared_dynamics_blocks = nn.Sequential(
            *[ResidualBlock(hidden_planes) for _ in range(num_blocks)]
        )

        # Per-game heads
        self.policy_heads = nn.ModuleDict()
        self.value_heads = nn.ModuleDict()
        self.reward_heads = nn.ModuleDict()
        for name, cfg in game_configs.items():
            self.policy_heads[name] = nn.Sequential(
                nn.Conv2d(hidden_planes, 2, 1, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2 * latent_h * latent_w, cfg["action_space"]),
            )
            self.value_heads[name] = nn.Sequential(
                nn.Conv2d(hidden_planes, 1, 1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Flatten(),
                mlp_head(latent_h * latent_w, fc_hidden, value_out),
            )
            self.reward_heads[name] = nn.Sequential(
                nn.Conv2d(hidden_planes, 1, 1),
                nn.Flatten(),
                mlp_head(latent_h * latent_w, fc_hidden, reward_out),
            )

    def initial_inference(self, obs: torch.Tensor, game_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proj = self.input_projections[game_name](obs)
        if proj.shape[-2:] != (self.latent_h, self.latent_w):
            proj = F.interpolate(proj, size=(self.latent_h, self.latent_w), mode="bilinear", align_corners=False)
        hidden = self.shared_repr_blocks(proj)
        hidden = _min_max_normalize(hidden)
        policy_logits = self.policy_heads[game_name](hidden)
        value_logits = self.value_heads[game_name](hidden)
        value = self._value_logits_to_scalar(value_logits)
        return hidden, policy_logits, value

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: torch.Tensor, game_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = hidden_state.shape[0]
        max_action = max(1, action.max().item() + 1)
        action_plane = action.float() / max_action
        action_plane = action_plane.view(b, 1, 1, 1).expand(b, 1, self.latent_h, self.latent_w)

        x = torch.cat([hidden_state, action_plane], dim=1)
        x = self.dynamics_conv_in(x)
        x = self.dynamics_bn_in(x)
        # Residual skip connection
        x = F.relu(x + hidden_state)
        x = self.shared_dynamics_blocks(x)
        x = _min_max_normalize(x)

        reward_logits = self.reward_heads[game_name](x)
        reward = self._reward_logits_to_scalar(reward_logits)
        policy_logits = self.policy_heads[game_name](x)
        value_logits = self.value_heads[game_name](x)
        value = self._value_logits_to_scalar(value_logits)
        return x, reward, policy_logits, value

    def initial_inference_logits(self, obs: torch.Tensor, game_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (hidden_state, policy_logits, value_logits) for training."""
        proj = self.input_projections[game_name](obs)
        if proj.shape[-2:] != (self.latent_h, self.latent_w):
            proj = F.interpolate(proj, size=(self.latent_h, self.latent_w), mode="bilinear", align_corners=False)
        hidden = self.shared_repr_blocks(proj)
        hidden = _min_max_normalize(hidden)
        policy_logits = self.policy_heads[game_name](hidden)
        value_logits = self.value_heads[game_name](hidden)
        return hidden, policy_logits, value_logits

    def recurrent_inference_logits(
        self, hidden_state: torch.Tensor, action: torch.Tensor, game_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (next_hidden, reward_logits, policy_logits, value_logits) for training."""
        b = hidden_state.shape[0]
        max_action = max(1, action.max().item() + 1)
        action_plane = action.float() / max_action
        action_plane = action_plane.view(b, 1, 1, 1).expand(b, 1, self.latent_h, self.latent_w)

        x = torch.cat([hidden_state, action_plane], dim=1)
        x = self.dynamics_conv_in(x)
        x = self.dynamics_bn_in(x)
        x = F.relu(x + hidden_state)
        x = self.shared_dynamics_blocks(x)
        x = _min_max_normalize(x)

        reward_logits = self.reward_heads[game_name](x)
        policy_logits = self.policy_heads[game_name](x)
        value_logits = self.value_heads[game_name](x)
        return x, reward_logits, policy_logits, value_logits

    def _value_logits_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        scalar = support_to_scalar(logits, self.value_support_size)
        return inverse_scalar_transform(scalar).unsqueeze(-1)

    def _reward_logits_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        scalar = support_to_scalar(logits, self.reward_support_size)
        return inverse_scalar_transform(scalar).unsqueeze(-1)


def _min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] per sample."""
    b = x.shape[0]
    x_flat = x.view(b, -1)
    x_min = x_flat.min(dim=1, keepdim=True).values
    x_max = x_flat.max(dim=1, keepdim=True).values
    denom = (x_max - x_min).clamp(min=1e-8)
    x_flat = (x_flat - x_min) / denom
    return x_flat.view_as(x)
