"""Multi-game MuZero training with shared backbone."""

import os
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import MuZeroConfig, get_config
from ..games import GAME_REGISTRY
from ..model.muzero_net import MultiGameMuZeroNetwork
from ..model.utils import scalar_transform, scalar_to_support
from .replay_buffer import ReplayBuffer
from .self_play import run_self_play


class MultiGameTrainer:
    """Train a single shared MuZero model across multiple games."""

    def __init__(
        self,
        game_names: list[str],
        run_id: str,
        hidden_planes: int = 128,
        num_blocks: int = 8,
        latent_size: int = 8,
        fc_hidden: int = 128,
        training_steps: int = 200000,
        device: str = "cpu",
        log_dir: str = "runs",
        checkpoints_dir: str = "checkpoints",
    ):
        self.device = device
        self.training_steps = training_steps
        self.game_names = game_names
        self.run_id = run_id
        self.checkpoint_dir = Path(checkpoints_dir) / "multi_game" / run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create games and configs
        self.games = {}
        self.configs = {}
        self.replay_buffers = {}
        for name in game_names:
            self.games[name] = GAME_REGISTRY[name]()
            self.configs[name] = get_config(name)
            self.replay_buffers[name] = ReplayBuffer(self.configs[name].replay_buffer_size)

        # Build multi-game network
        game_configs = {}
        for name in game_names:
            game = self.games[name]
            game_configs[name] = {
                "obs_channels": game.num_planes,
                "action_space": game.action_space_size,
                "input_h": game.board_size[0],
                "input_w": game.board_size[1],
            }

        self.network = MultiGameMuZeroNetwork(
            game_configs=game_configs,
            hidden_planes=hidden_planes,
            num_blocks=num_blocks,
            latent_h=latent_size,
            latent_w=latent_size,
            fc_hidden=fc_hidden,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=1e-3, weight_decay=1e-4
        )
        use_amp = device.startswith("cuda")
        self.scaler = GradScaler("cuda", enabled=use_amp)
        self.use_amp = use_amp

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "multi_game", run_id))
        self.global_step = 0

        print(f"Multi-game MuZero: {game_names}")
        print(f"Parameters: {sum(p.numel() for p in self.network.parameters()):,}")

    def train(self):
        """Run multi-game training with round-robin sampling."""
        # Initial self-play for all games
        for name in self.game_names:
            print(f"Generating initial self-play for {name}...")
            self._run_self_play(name)

        game_cycle = cycle(self.game_names)
        pbar = tqdm(range(self.training_steps), desc="Multi-game training")

        for step in pbar:
            self.global_step = step
            game_name = next(game_cycle)

            # Periodic self-play
            cfg = self.configs[game_name]
            if step > 0 and step % cfg.self_play_interval == 0:
                self._run_self_play(game_name)

            # Training step for current game
            loss_info = self._train_step(game_name)

            if step % 50 == 0:
                pbar.set_postfix({"game": game_name, "loss": f"{loss_info['total_loss']:.4f}"})
                for k, v in loss_info.items():
                    self.writer.add_scalar(f"loss/{game_name}/{k}", v, step)

            if step > 0 and step % 1000 == 0:
                self._save_checkpoint(step)

            if step > 0 and step % 2000 == 0:
                for name in self.game_names:
                    self._evaluate(name, step)

        self._save_checkpoint(self.training_steps)
        self.writer.close()

    def _run_self_play(self, game_name: str):
        """Run self-play for a specific game using the multi-game network."""
        game = self.games[game_name]
        config = self.configs[game_name]

        # Create a wrapper that routes through the multi-game network
        wrapper = _MultiGameNetWrapper(self.network, game_name)
        games = run_self_play(wrapper, game, config,
                              config.num_self_play_games, self.device, show_progress=False)
        for g in games:
            g.game_name = game_name
            self.replay_buffers[game_name].save_game(g)

    def _train_step(self, game_name: str) -> dict:
        self.network.train()
        game = self.games[game_name]
        config = self.configs[game_name]
        buffer = self.replay_buffers[game_name]

        if len(buffer) < config.min_buffer_size:
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "reward_loss": 0.0}

        batch = buffer.sample_batch(
            config.batch_size, config.num_unroll_steps,
            config.td_steps, config.discount, game.action_space_size,
        )

        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        target_values = batch["target_values"].to(self.device)
        target_rewards = batch["target_rewards"].to(self.device)
        target_policies = batch["target_policies"].to(self.device)

        value_support = self.network.value_support_size
        reward_support = self.network.reward_support_size
        value_scale = getattr(self.network, "value_target_scale", 1.0)

        with autocast(device_type=self.device.split(":")[0], enabled=self.use_amp):
            hidden, policy_logits, value_logits = self.network.initial_inference_logits(obs, game_name)

            policy_loss = -(target_policies[:, 0] * F.log_softmax(policy_logits, dim=1)).sum(1).mean()
            value_loss = self._categorical_loss(value_logits, target_values[:, 0], value_support, value_scale)
            reward_loss = torch.tensor(0.0, device=self.device)

            for k in range(config.num_unroll_steps):
                hidden, reward_logits, policy_logits, value_logits = \
                    self.network.recurrent_inference_logits(hidden, actions[:, k], game_name)
                hidden.register_hook(lambda grad: grad * 0.5)

                policy_loss += -(target_policies[:, k+1] * F.log_softmax(policy_logits, dim=1)).sum(1).mean()
                value_loss += self._categorical_loss(value_logits, target_values[:, k+1], value_support, value_scale)
                reward_loss += self._categorical_loss(reward_logits, target_rewards[:, k+1], reward_support)

            scale = 1.0 / (config.num_unroll_steps + 1)
            total_loss = scale * (
                policy_loss
                + config.value_loss_weight * value_loss
                + reward_loss
            )

        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": (scale * policy_loss).item(),
            "value_loss": (scale * config.value_loss_weight * value_loss).item(),
            "reward_loss": (scale * reward_loss).item(),
        }

    def _categorical_loss(self, logits: torch.Tensor, target_scalar: torch.Tensor,
                          support_size: int, target_scale: float = 1.0) -> torch.Tensor:
        """Cross-entropy loss for categorical value/reward prediction."""
        if getattr(self.network, "use_scalar_transform", True):
            transformed = scalar_transform(target_scalar)
        else:
            transformed = target_scalar * target_scale
        target_dist = scalar_to_support(transformed, support_size).to(logits.device)
        return -(target_dist * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    @torch.no_grad()
    def _evaluate(self, game_name: str, step: int):
        import random
        self.network.eval()
        game = self.games[game_name]
        config = self.configs[game_name]
        wrapper = _MultiGameNetWrapper(self.network, game_name)

        from ..mcts.mcts import MCTS, select_action
        mcts = MCTS(wrapper, game, config, self.device)

        wins = 0
        num_games = 50
        for _ in range(num_games):
            state = game.reset()
            while not state.done:
                legal = game.legal_actions(state)
                if state.current_player == 1:
                    obs = game.to_tensor(state)
                    root = mcts.run(obs, legal, add_noise=False)
                    action, _ = select_action(root, temperature=0)
                else:
                    action = random.choice(legal)
                state, _, _ = game.step(state, action)
            if state.winner == 1:
                wins += 1

        win_rate = wins / num_games
        self.writer.add_scalar(f"eval/{game_name}/win_rate_vs_random", win_rate, step)
        tqdm.write(f"Step {step} [{game_name}]: {win_rate:.1%} win rate vs random")

    def _save_checkpoint(self, step: int):
        torch.save({
            "step": step,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "game_names": self.game_names,
        }, self.checkpoint_dir / f"checkpoint_{step}.pt")


class _MultiGameNetWrapper:
    """Wraps MultiGameMuZeroNetwork to look like a single-game MuZeroNetwork for MCTS."""

    def __init__(self, network: MultiGameMuZeroNetwork, game_name: str):
        self.network = network
        self.game_name = game_name

    def initial_inference(self, obs):
        return self.network.initial_inference(obs, self.game_name)

    def recurrent_inference(self, hidden_state, action):
        return self.network.recurrent_inference(hidden_state, action, self.game_name)

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()
