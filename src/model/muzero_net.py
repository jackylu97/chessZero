"""MuZero neural networks: representation, dynamics, prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    ResidualBlock,
    inverse_scalar_transform,
    mlp_head,
    norm_layer,
    support_to_scalar,
    wdl_to_scalar,
)


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

        # Projection norm operates at the raw input resolution (may differ from
        # latent if needs_resize); residual blocks operate at the latent size.
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_planes, 3, padding=1, bias=False),
            norm_layer(hidden_planes, (input_h, input_w)),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_planes, (latent_h, latent_w)) for _ in range(num_blocks)]
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

    Action is encoded as a learned ``nn.Embedding`` of dimension ``action_embed_dim``,
    broadcast spatially as that many channels and concatenated with the hidden state.
    Includes a residual skip from the input hidden state matching LightZero's
    DynamicsNetwork (lzero/model/muzero_model.py:519-525, "the residual link").

    Why a learned embedding over the prior scalar broadcast (action / action_space_size,
    1 channel)?
    The conv_in input channel for the action gets a single weight slice in
    ``conv_in.weight``, so the dynamics output across different actions can only
    differ by a scalar magnitude — architecturally incapable of producing
    qualitatively different transformations per action. Probed on the
    2026_05_07_small_cold checkpoint (step 1000), this manifested as
    ``cos(dynamics(h, a_i), dynamics(h, a_j)) = 0.9999`` across 20 distinct chess
    moves: action-blind dynamics, flat Q values across MCTS, policy collapses to
    "do-nothing" attractors (king shuffles → 3-fold repetition).
    A learned ``nn.Embedding(action_space_size, embed_dim)`` gives ``conv_in`` a
    full ``embed_dim`` × hidden_planes × 9 weight block to encode per-action
    transformations — closer to LightZero's one-hot (4672 channels for chess) but
    without paying the full one-hot cost. ``embed_dim=16`` is the configurable
    default; ``embed_dim=action_space_size`` recovers (un-trained) one-hot.

    Fallback: if learned embedding still under-discriminates after a sanity-check
    run, switch to LightZero-style one-hot by setting ``action_embed_dim`` to
    ``action_space_size`` in the config (or implement the one-hot path explicitly
    if the lookup-table init becomes a bottleneck).
    """

    def __init__(
        self,
        hidden_planes: int,
        num_blocks: int,
        action_space_size: int,
        latent_h: int,
        latent_w: int,
        fc_hidden: int,
        action_embed_dim: int = 16,
        reward_support_size: int = 1,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.action_embed_dim = action_embed_dim
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.reward_support_size = reward_support_size

        # Learned per-action embedding broadcast spatially. Default N(0, 1) init.
        self.action_embedding = nn.Embedding(action_space_size, action_embed_dim)

        # Conv consumes hidden_state + action embedding (embed_dim channels).
        self.conv_in = nn.Conv2d(
            hidden_planes + action_embed_dim, hidden_planes, 3, padding=1, bias=False
        )
        self.bn_in = norm_layer(hidden_planes, (latent_h, latent_w))

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_planes, (latent_h, latent_w)) for _ in range(num_blocks)]
        )

        # Reward head outputs categorical distribution
        reward_out = 2 * reward_support_size + 1
        self.reward_head = nn.Sequential(
            nn.Conv2d(hidden_planes, 1, 1),
            nn.Flatten(),
            mlp_head(latent_h * latent_w, fc_hidden, reward_out),
        )
        _zero_init_last_linear(self.reward_head)

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
        # Lookup learned per-action embedding, then broadcast spatially.
        action_embed = self.action_embedding(action.long())              # (B, embed_dim)
        action_plane = action_embed.view(b, self.action_embed_dim, 1, 1)
        action_plane = action_plane.expand(b, self.action_embed_dim, self.latent_h, self.latent_w)

        x = torch.cat([hidden_state, action_plane], dim=1)
        x = self.conv_in(x)
        x = self.bn_in(x)
        # Residual skip connection from input hidden state (matches LightZero).
        x = F.relu(x + hidden_state)
        x = self.blocks(x)
        x = _min_max_normalize(x)

        reward_logits = self.reward_head(x)
        return x, reward_logits


class ConvPolicyHead(nn.Module):
    """AlphaZero-style spatial policy head.

    For board games whose action space factorizes as
    ``(latent_h * latent_w)`` from-squares × ``move_planes`` move-types, with
    flat index ``from_sq * move_planes + move_type`` (chess: 64 × 73 = 4672).

    A 3×3 conv mixes the body's FULL-width per-square features (no channel
    bottleneck — contrast the flat head's ``Conv(C->2)`` squeeze), then a 1×1
    conv projects to ``move_planes`` logits per square. The projection weights
    are SHARED across all from-squares (the right inductive bias: a move-type's
    value is computed the same way wherever the piece sits). Board-wide context
    is already supplied by the deep conv body (receptive field >> 8×8); this
    head only needs the per-square readout, so a 1×1 projection suffices.

    The ``(B, move_planes, H, W)`` output is permuted to ``(B, H, W, move_planes)``
    and flattened so the logit index equals ``from_sq * move_planes + move_type``,
    matching the game's action encoding exactly — no change to legal masks,
    policy targets, or action decoding. Requires the latent's spatial layout to
    match ``from_sq = row * latent_w + col`` (true for the 8×8 chess latent);
    gate behind ``policy_head_type="conv"`` for chess only.
    """

    def __init__(self, hidden_planes: int, action_space_size: int,
                 latent_h: int, latent_w: int):
        super().__init__()
        n_sq = latent_h * latent_w
        if action_space_size % n_sq != 0:
            raise ValueError(
                f"ConvPolicyHead needs action_space_size ({action_space_size}) "
                f"divisible by latent_h*latent_w ({n_sq}); got remainder "
                f"{action_space_size % n_sq}."
            )
        self.action_space_size = action_space_size
        self.move_planes = action_space_size // n_sq
        self.mix = nn.Conv2d(hidden_planes, hidden_planes, 3, padding=1, bias=False)
        self.norm = norm_layer(hidden_planes, (latent_h, latent_w))
        self.proj = nn.Conv2d(hidden_planes, self.move_planes, 1)
        # Zero-init the output conv -> uniform policy at start (the MuZero
        # last-layer-zero stability trick; mirrors _zero_init_last_linear on the
        # flat head).
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm(self.mix(x)))
        x = self.proj(x)                                  # (B, move_planes, H, W)
        b = x.shape[0]
        # (B, P, H, W) -> (B, H, W, P) -> (B, H*W*P); index = from_sq*P + move_type
        return x.permute(0, 2, 3, 1).reshape(b, self.action_space_size)


class PredictionNetwork(nn.Module):
    """f(hidden_state) -> (policy, value).

    Two value head types are supported:

    - ``value_head_type="support"`` (default, MuZero paper): output a categorical
      distribution over a discrete support {-K, ..., 0, ..., +K}. Decoded to a
      scalar via ``support_to_scalar``. Width = ``2 * value_support_size + 1``.

    - ``value_head_type="wdl"`` (Lc0 production): output 3 logits over
      (Win, Draw, Loss) from the side-to-move's POV. Decoded to a scalar via
      ``wdl_to_scalar`` (V = P(W) - P(L) + draw_score · P(D)).
    """

    def __init__(
        self,
        hidden_planes: int,
        action_space_size: int,
        latent_h: int,
        latent_w: int,
        fc_hidden: int,
        value_support_size: int = 1,
        value_head_type: str = "support",
        value_head_init_std: float = 0.0,
        policy_head_type: str = "flat",
    ):
        super().__init__()
        self.value_support_size = value_support_size
        self.value_head_type = value_head_type
        self.policy_head_type = policy_head_type

        # Policy head. "flat" (default, AlphaGo-style): a Conv(C->2) channel
        # squeeze + Linear to the full action space — fine for small/point-move
        # action spaces but a severe bottleneck for chess's structured 4672.
        # "conv" (AlphaZero-style): a spatial 73-plane conv head with no channel
        # squeeze and weight-sharing across from-squares (see ConvPolicyHead).
        if policy_head_type == "conv":
            self.policy_head = ConvPolicyHead(
                hidden_planes, action_space_size, latent_h, latent_w
            )  # zero-inits its own output conv
        elif policy_head_type == "flat":
            self.policy_head = nn.Sequential(
                nn.Conv2d(hidden_planes, 2, 1, bias=False),
                norm_layer(2, (latent_h, latent_w)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2 * latent_h * latent_w, action_space_size),
            )
            _zero_init_last_linear(self.policy_head)
        else:
            raise ValueError(f"Unknown policy_head_type: {policy_head_type!r}")

        # Value head: shape depends on the head type.
        if value_head_type == "support":
            value_out = 2 * value_support_size + 1
        elif value_head_type == "wdl":
            value_out = 3
        else:
            raise ValueError(f"Unknown value_head_type: {value_head_type!r}")
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_planes, 1, 1, bias=False),
            norm_layer(1, (latent_h, latent_w)),
            nn.ReLU(),
            nn.Flatten(),
            mlp_head(latent_h * latent_w, fc_hidden, value_out),
        )
        # Value-head output init. Default (std=0.0) zero-inits the last linear —
        # the standard MuZero/LightZero stability trick. BUT zero weights make the
        # head's Jacobian w.r.t. its input zero (grad_to_input = Wᵀ·grad_out = 0),
        # so at cold start the value head passes NO gradient back to the
        # representation/dynamics body — only the head's own weights move. A small
        # nonzero std lets some body gradient flow from step 0 (removes the cold-
        # start block; does NOT fix draw-basin gradient vanishing — that needs a
        # body-direct signal like inverse dynamics). See dynamics_gradient_starvation.
        if value_head_init_std and value_head_init_std > 0:
            _small_init_last_linear(self.value_head, value_head_init_std)
        else:
            _zero_init_last_linear(self.value_head)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            policy_logits: (B, action_space_size)
            value_logits:
              - support head: (B, 2*support_size+1)
              - wdl head:     (B, 3)  — order (W, D, L)
        """
        return self.policy_head(hidden_state), self.value_head(hidden_state)


class ProjectionNetwork(nn.Module):
    """SimSiam projector: flattened latent → proj_out.

    Three-layer MLP (Linear+BN+ReLU twice, then Linear+BN). Matches LightZero's
    EfficientZero projection: lzero/model/efficientzero_model.py.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHead(nn.Module):
    """SimSiam predictor: proj_out → pred_out.

    Two-layer MLP (Linear+BN+ReLU, Linear). Asymmetry with the projector is
    the key ingredient that prevents representation collapse (Chen & He 2020).
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InverseDynamicsHead(nn.Module):
    """Predict action a_k from (h_k, h_{k+1}) — ICM/Pathak inverse model.

    The action is recoverable from (h_k, dynamics(h_k, a_k)) ONLY if the dynamics
    output actually depends on a_k. Minimizing the inverse-prediction loss is
    therefore a non-bypassable pressure forcing the dynamics to encode the action
    — directly counteracting the action-blind collapse where dynamics(h, a_i) ≈
    dynamics(h, a_j) for all actions.

    Unlike the EfficientZero consistency loss (which trains dynamics(h,a)→repr(next)
    and is satisfiable by memorizing h→next while ignoring a), the inverse model
    cannot be satisfied by an action-blind dynamics. Verified on the chess preset:
    inverse loss drives cross-action cos 1.0→0.61 where every consistency variant
    (8-frame / single-frame / contrastive) stayed at ~1.0. See
    dynamics_gradient_starvation memory note + scripts/probe_fix_candidates.py.
    """

    def __init__(self, hidden_planes: int, latent_h: int, latent_w: int,
                 action_space_size: int, hidden: int = 256):
        super().__init__()
        in_dim = 2 * hidden_planes * latent_h * latent_w
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, action_space_size),
        )

    def forward(self, h_k: torch.Tensor, h_next: torch.Tensor) -> torch.Tensor:
        b = h_k.shape[0]
        x = torch.cat([h_k.reshape(b, -1), h_next.reshape(b, -1)], dim=1)
        return self.net(x)


class MovesLeftHead(nn.Module):
    """Lc0-style moves-left head: predicts plies-remaining-until-game-end as a
    categorical distribution over the transformed support {-K..K} (the value head's
    machinery; only the non-negative half is used in practice).

    Purpose: break value *saturation* in won/lost positions. When every winning move
    reads ~+1 (or, mid-collapse, ~draw), the value gives MCTS no reason to make
    progress and it shuffles. Moves-left restores a gradient: among equally-winning
    moves prefer the faster finish (and drag out a loss). POV-independent — game
    length is the same for both players, so unlike value there is NO negamax flip.

    Trained via cross-entropy on scalar_to_support(scalar_transform(plies_left)).
    Queried separately (like the inverse/consistency heads), so the main inference
    tuple signature is unchanged.
    """

    def __init__(self, hidden_planes: int, latent_h: int, latent_w: int,
                 fc_hidden: int, support_size: int):
        super().__init__()
        self.support_size = support_size
        self.head = nn.Sequential(
            nn.Conv2d(hidden_planes, 1, 1, bias=False),
            norm_layer(1, (latent_h, latent_w)),
            nn.ReLU(),
            nn.Flatten(),
            mlp_head(latent_h * latent_w, fc_hidden, 2 * support_size + 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_state)  # (B, 2*support_size + 1) logits


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
        action_embed_dim: int = 16,
        use_consistency_loss: bool = False,
        proj_hid: int = 1024,
        proj_out: int = 1024,
        pred_hid: int = 512,
        pred_out: int = 1024,
        use_scalar_transform: bool = True,
        value_target_scale: float = 1.0,
        value_head_type: str = "support",
        draw_score: float = 0.0,
        value_head_init_std: float = 0.0,
        use_inverse_dynamics_loss: bool = False,
        inverse_dynamics_hidden: int = 256,
        policy_head_type: str = "flat",
        use_moves_left: bool = False,
        moves_left_support_size: int = 10,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.value_support_size = value_support_size
        self.reward_support_size = reward_support_size
        self.use_consistency_loss = use_consistency_loss
        self.use_scalar_transform = use_scalar_transform
        self.value_target_scale = value_target_scale
        self.value_head_type = value_head_type
        self.draw_score = draw_score
        self.use_inverse_dynamics_loss = use_inverse_dynamics_loss
        self.use_moves_left = use_moves_left
        self.moves_left_support_size = moves_left_support_size

        self.representation = RepresentationNetwork(
            observation_channels, hidden_planes, num_blocks,
            latent_h, latent_w, input_h, input_w,
        )
        self.dynamics = DynamicsNetwork(
            hidden_planes, num_blocks, action_space_size,
            latent_h, latent_w, fc_hidden,
            action_embed_dim=action_embed_dim,
            reward_support_size=reward_support_size,
        )
        self.prediction = PredictionNetwork(
            hidden_planes, action_space_size,
            latent_h, latent_w, fc_hidden, value_support_size,
            value_head_type=value_head_type,
            value_head_init_std=value_head_init_std,
            policy_head_type=policy_head_type,
        )

        # Inverse-dynamics head (ICM): predicts a_k from (h_k, h_{k+1}). Forces the
        # dynamics output to encode the action. Training-only; not used by MCTS.
        if use_inverse_dynamics_loss:
            self.inverse_dynamics_head = InverseDynamicsHead(
                hidden_planes, latent_h, latent_w, action_space_size,
                hidden=inverse_dynamics_hidden,
            )
        else:
            self.inverse_dynamics_head = None

        # Moves-left head (Lc0): predicts plies-to-end; used to break value
        # saturation so the search prefers faster wins instead of shuffling.
        if use_moves_left:
            self.moves_left_head = MovesLeftHead(
                hidden_planes, latent_h, latent_w, fc_hidden, moves_left_support_size,
            )
        else:
            self.moves_left_head = None

        if use_consistency_loss:
            flat_dim = hidden_planes * latent_h * latent_w
            self.projection = ProjectionNetwork(flat_dim, proj_hid, proj_out)
            self.prediction_head = PredictionHead(proj_out, pred_hid, pred_out)
        else:
            self.projection = None
            self.prediction_head = None

    def project(self, hidden: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        """SimSiam projection for EfficientZero consistency loss.

        with_grad=True:  predictor(projection(hidden))  — online branch (gradient flows)
        with_grad=False: projection(hidden).detach()    — target branch (stop-grad)

        Matches LightZero's MuZeroModel.project() signature.
        """
        assert self.projection is not None, "project() called with use_consistency_loss=False"
        x = hidden.reshape(hidden.shape[0], -1)
        proj = self.projection(x)
        if with_grad:
            return self.prediction_head(proj)
        return proj.detach()

    def predict_inverse_action(self, h_k: torch.Tensor, h_next: torch.Tensor) -> torch.Tensor:
        """Inverse-dynamics: logits over actions predicted from (h_k, h_{k+1}).

        Gradient flows into BOTH hidden states (no detach) so the dynamics learns
        to make its output action-recoverable. Training-only.
        """
        assert self.inverse_dynamics_head is not None, \
            "predict_inverse_action() called with use_inverse_dynamics_loss=False"
        return self.inverse_dynamics_head(h_k, h_next)

    def predict_moves_left(self, hidden: torch.Tensor) -> torch.Tensor:
        """Moves-left logits (B, 2*K+1) from a hidden state. Queried separately
        (training: CE loss; search: gated utility). POV-independent."""
        assert self.moves_left_head is not None, \
            "predict_moves_left() called with use_moves_left=False"
        return self.moves_left_head(hidden)

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
        if getattr(self, "value_head_type", "support") == "wdl":
            return wdl_to_scalar(logits, draw_score=getattr(self, "draw_score", 0.0)).unsqueeze(-1)
        scalar = support_to_scalar(logits, self.value_support_size)
        if self.use_scalar_transform:
            scalar = inverse_scalar_transform(scalar)
        elif self.value_target_scale != 1.0:
            scalar = scalar / self.value_target_scale
        return scalar.unsqueeze(-1)

    def _reward_logits_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        scalar = support_to_scalar(logits, self.reward_support_size)
        if self.use_scalar_transform:
            scalar = inverse_scalar_transform(scalar)
        return scalar.unsqueeze(-1)


class MultiGameMuZeroNetwork(nn.Module):
    """Multi-game MuZero with shared backbone and game-specific heads.

    Phase 2: Shared residual backbone processes all games.
    Per-game input projections and output heads handle different
    observation/action spaces.
    """

    def __init__(self, game_configs: dict, hidden_planes: int = 128,
                 num_blocks: int = 8, latent_h: int = 8, latent_w: int = 8,
                 fc_hidden: int = 128, game_id_dim: int = 16,
                 value_support_size: int = 10, reward_support_size: int = 1,
                 use_scalar_transform: bool = True,
                 value_target_scale: float = 1.0):
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
        self.use_scalar_transform = use_scalar_transform
        self.value_target_scale = value_target_scale

        value_out = 2 * value_support_size + 1
        reward_out = 2 * reward_support_size + 1

        # Game ID embeddings — defined but not yet wired up.
        # To use: concatenate the embedding for the current game onto the hidden state
        # before the shared backbone, so the network has an explicit "which game am I playing"
        # signal. This may improve multi-game alignment if shared training underperforms.
        self.game_embeddings = nn.Embedding(len(self.game_names), game_id_dim)

        # Per-game input projections (each at the game's native input resolution,
        # before padding to the shared latent size).
        self.input_projections = nn.ModuleDict()
        for name, cfg in game_configs.items():
            self.input_projections[name] = nn.Sequential(
                nn.Conv2d(cfg["obs_channels"], hidden_planes, 3, padding=1, bias=False),
                norm_layer(hidden_planes, (cfg["input_h"], cfg["input_w"])),
                nn.ReLU(),
            )

        # Shared backbone (representation) — runs at (latent_h, latent_w) after padding.
        self.shared_repr_blocks = nn.Sequential(
            *[ResidualBlock(hidden_planes, (latent_h, latent_w)) for _ in range(num_blocks)]
        )

        # Shared dynamics backbone
        self.dynamics_conv_in = nn.Conv2d(hidden_planes + 1, hidden_planes, 3, padding=1, bias=False)
        self.dynamics_bn_in = norm_layer(hidden_planes, (latent_h, latent_w))
        self.shared_dynamics_blocks = nn.Sequential(
            *[ResidualBlock(hidden_planes, (latent_h, latent_w)) for _ in range(num_blocks)]
        )

        # Per-game heads
        self.policy_heads = nn.ModuleDict()
        self.value_heads = nn.ModuleDict()
        self.reward_heads = nn.ModuleDict()
        for name, cfg in game_configs.items():
            self.policy_heads[name] = nn.Sequential(
                nn.Conv2d(hidden_planes, 2, 1, bias=False),
                norm_layer(2, (latent_h, latent_w)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2 * latent_h * latent_w, cfg["action_space"]),
            )
            _zero_init_last_linear(self.policy_heads[name])

            self.value_heads[name] = nn.Sequential(
                nn.Conv2d(hidden_planes, 1, 1, bias=False),
                norm_layer(1, (latent_h, latent_w)),
                nn.ReLU(),
                nn.Flatten(),
                mlp_head(latent_h * latent_w, fc_hidden, value_out),
            )
            _zero_init_last_linear(self.value_heads[name])

            self.reward_heads[name] = nn.Sequential(
                nn.Conv2d(hidden_planes, 1, 1),
                nn.Flatten(),
                mlp_head(latent_h * latent_w, fc_hidden, reward_out),
            )
            _zero_init_last_linear(self.reward_heads[name])

    def initial_inference(self, obs: torch.Tensor, game_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proj = self.input_projections[game_name](obs)
        if proj.shape[-2:] != (self.latent_h, self.latent_w):
            proj = _pad_to_size(proj, self.latent_h, self.latent_w)
        hidden = self.shared_repr_blocks(proj)
        hidden = _min_max_normalize(hidden)
        policy_logits = self.policy_heads[game_name](hidden)
        value_logits = self.value_heads[game_name](hidden)
        value = self._value_logits_to_scalar(value_logits)
        return hidden, policy_logits, value

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: torch.Tensor, game_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO(2026-05-07, multi-game): port the per-game ``nn.Embedding`` action
        # encoding from DynamicsNetwork. The scalar broadcast below has the same
        # action-blindness pathology that motivated the DynamicsNetwork rewrite
        # (cos(dynamics(h, a_i), dynamics(h, a_j))≈1.0). Multi-game training is
        # currently dormant so the bug is not blocking; fix before re-enabling.
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
            proj = _pad_to_size(proj, self.latent_h, self.latent_w)
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
        if getattr(self, "value_head_type", "support") == "wdl":
            return wdl_to_scalar(logits, draw_score=getattr(self, "draw_score", 0.0)).unsqueeze(-1)
        scalar = support_to_scalar(logits, self.value_support_size)
        if self.use_scalar_transform:
            scalar = inverse_scalar_transform(scalar)
        elif self.value_target_scale != 1.0:
            scalar = scalar / self.value_target_scale
        return scalar.unsqueeze(-1)

    def _reward_logits_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        scalar = support_to_scalar(logits, self.reward_support_size)
        if self.use_scalar_transform:
            scalar = inverse_scalar_transform(scalar)
        return scalar.unsqueeze(-1)


def _min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize hidden state to [0, 1] per (sample, channel).

    Normalizes each channel independently rather than globally across all channels.
    This preserves relative scale differences between channels — e.g. a channel
    encoding piece count and a channel encoding mobility can have different ranges
    and both get correctly scaled. Matches the muzero-general ResNet implementation
    (see models.py representation/dynamics methods).

    Previous implementation normalized globally per sample (view(b, -1)), which
    collapsed all inter-channel scale information into a single [0,1] range.
    """
    b, c = x.shape[0], x.shape[1]
    x_flat = x.view(b, c, -1)                                  # (B, C, H*W)
    x_min = x_flat.min(dim=2, keepdim=True).values             # (B, C, 1)
    x_max = x_flat.max(dim=2, keepdim=True).values             # (B, C, 1)
    denom = (x_max - x_min).clamp(min=1e-5)
    x_flat = (x_flat - x_min) / denom
    return x_flat.view_as(x)


def _zero_init_last_linear(module: nn.Module) -> None:
    """Zero-initialize the last Linear layer in a module or nested Sequential.

    Initializing output layers to zero means all heads predict zero at the start
    of training. This avoids large initial gradients from random predictions and
    improves early convergence stability. Used by LightZero (last_linear_layer_init_zero)
    and recommended practice for value/policy/reward heads in MuZero variants.
    """
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Sequential):
        _zero_init_last_linear(module[-1])


def _small_init_last_linear(module: nn.Module, std: float) -> None:
    """Small-variance normal init of the last Linear (bias zeroed).

    Used as a flag-gated alternative to ``_zero_init_last_linear`` for the value
    head: keeps initial output magnitude tiny (near-agnostic predictions, MCTS
    stable) while making the head's Jacobian w.r.t. its input NONZERO, so gradient
    reaches the representation/dynamics body from training step 0 instead of being
    blocked until the zero-init weights drift away from zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Sequential):
        _small_init_last_linear(module[-1], std)


def _pad_to_size(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Zero-pad a spatial tensor (B, C, H, W) to (B, C, target_h, target_w).

    Padding is distributed evenly on both sides; if the difference is odd the
    extra row/col goes on the right/bottom.

    Why padding instead of bilinear interpolation:
        Bilinear rescaling produces non-integer blended values that distort the
        discrete piece-identity semantics of board games (e.g. Connect4 6x7 ->
        8x8 via bilinear creates fractional "half-pieces"). Zero-padding
        preserves every original cell exactly, and the network learns to ignore
        the constant-zero border.

    Alternatives considered:
        1. Per-task encoder -> flat latent vector (ScaleZero 2025, current SOTA):
           Each game's encoder maps its board to a fixed-size 1D vector; shared
           dynamics operates on vectors rather than spatial tensors. Avoids the
           alignment problem entirely but sacrifices the spatial inductive bias
           of convolutions in the shared backbone.

        2. Convert all game boards to native 8x8 (e.g. extend Connect4 to 8
           columns, play tic-tac-toe on an 8x8 board with standard rules):
           Cleanest option for the shared backbone — no padding artefacts — but
           changes game semantics and requires rewriting game logic.

        3. ViT / GNN backbone: patch tokenisation or graph message-passing
           naturally handles variable board sizes without any spatial alignment.
           Best choice if cross-size generalisation within a game family is a
           goal (AlphaViT 2024, ScalableAlphaZero 2021).

    Current choice: zero-padding to target_h x target_w (default 8x8).
    Revisit if multi-game training underperforms single-game baselines.
    """
    h, w = x.shape[-2], x.shape[-1]
    pad_h = target_h - h
    pad_w = target_w - w
    # F.pad format: (left, right, top, bottom)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
