"""MuZero training loop."""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import MuZeroConfig
from ..games.base import Game
from ..model.muzero_net import MuZeroNetwork
from ..model.utils import scalar_transform, scalar_to_support
from .replay_buffer import ReplayBuffer
from .self_play import run_self_play


class MuZeroTrainer:
    """Main training loop for MuZero."""

    def __init__(
        self,
        config: MuZeroConfig,
        game: Game,
        network: MuZeroNetwork,
        device: str = "cpu",
        log_dir: str = "runs",
    ):
        self.config = config
        self.game = game
        self.network = network.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scaler = GradScaler("cuda", enabled=config.use_amp)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, config.game))
        self.global_step = 0

    def train(self):
        """Run the full training loop."""
        print(f"Starting MuZero training for {self.config.game}")
        print(f"Device: {self.device}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")

        # Initial self-play to populate buffer
        print("Generating initial self-play games...")
        self._run_self_play()

        pbar = tqdm(range(self.config.training_steps), desc="Training")
        for step in pbar:
            self.global_step = step

            # Periodically generate new self-play games
            if step > 0 and step % self.config.self_play_interval == 0:
                self._run_self_play()

            # Training step
            loss_info = self._train_step()

            # Logging
            if step % self.config.log_interval == 0:
                self._log(loss_info, step)
                pbar.set_postfix({
                    "loss": f"{loss_info['total_loss']:.4f}",
                    "buf": len(self.replay_buffer),
                })

            # Checkpoint
            if step > 0 and step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(step)

            # Evaluation
            if step > 0 and step % self.config.eval_interval == 0:
                self._evaluate(step)

        self._save_checkpoint(self.config.training_steps)
        self.writer.close()
        print("Training complete!")

    def _run_self_play(self):
        """Generate self-play games and add to replay buffer."""
        games = run_self_play(
            self.network, self.game, self.config,
            self.config.num_self_play_games, self.device,
        )
        for g in games:
            self.replay_buffer.save_game(g)

        # Log self-play stats
        avg_length = sum(len(g) - 1 for g in games) / len(games)
        outcomes = [g.game_outcome for g in games]
        p1_wins = sum(1 for o in outcomes if o == 1)
        p2_wins = sum(1 for o in outcomes if o == -1)
        draws = sum(1 for o in outcomes if o == 0)

        self.writer.add_scalar("self_play/avg_game_length", avg_length, self.global_step)
        self.writer.add_scalar("self_play/p1_win_rate", p1_wins / len(games), self.global_step)
        self.writer.add_scalar("self_play/p2_win_rate", p2_wins / len(games), self.global_step)
        self.writer.add_scalar("self_play/draw_rate", draws / len(games), self.global_step)
        self.writer.add_scalar("self_play/buffer_size", len(self.replay_buffer), self.global_step)

    def _train_step(self) -> dict:
        """Single training step on a batch from replay buffer."""
        self.network.train()

        batch = self.replay_buffer.sample_batch(
            self.config.batch_size,
            self.config.num_unroll_steps,
            self.config.td_steps,
            self.config.discount,
            self.game.action_space_size,
        )

        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        target_values = batch["target_values"].to(self.device)
        target_rewards = batch["target_rewards"].to(self.device)
        target_policies = batch["target_policies"].to(self.device)

        with autocast(device_type=self.device.split(":")[0], enabled=self.config.use_amp):
            # Initial inference (use logits variant for training)
            hidden, policy_logits, value_logits = self.network.initial_inference_logits(obs)

            # Loss at root position (k=0)
            policy_loss = self._policy_loss(policy_logits, target_policies[:, 0])
            value_loss = self._value_loss(value_logits, target_values[:, 0],
                                          self.network.value_support_size)
            reward_loss = torch.tensor(0.0, device=self.device)  # no reward at root

            # Unroll K steps
            for k in range(self.config.num_unroll_steps):
                hidden, reward_logits, policy_logits, value_logits = \
                    self.network.recurrent_inference_logits(hidden, actions[:, k])

                # Scale gradient for dynamics network (MuZero paper Section 3)
                hidden.register_hook(lambda grad: grad * 0.5)

                policy_loss += self._policy_loss(policy_logits, target_policies[:, k + 1])
                value_loss += self._value_loss(value_logits, target_values[:, k + 1],
                                               self.network.value_support_size)
                reward_loss += self._reward_loss(reward_logits, target_rewards[:, k + 1],
                                                 self.network.reward_support_size)

            # Scale by 1/K and apply value loss weight (0.25 following Reanalyze paper)
            scale = 1.0 / (self.config.num_unroll_steps + 1)
            total_loss = scale * (
                policy_loss
                + self.config.value_loss_weight * value_loss
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
            "value_loss": (scale * self.config.value_loss_weight * value_loss).item(),
            "reward_loss": (scale * reward_loss).item(),
        }

    def _policy_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss between predicted policy and MCTS visit distribution."""
        return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def _value_loss(self, logits: torch.Tensor, target_scalar: torch.Tensor,
                    support_size: int) -> torch.Tensor:
        """Cross-entropy loss for categorical value prediction.

        Target scalars are transformed and projected onto the discrete support.
        """
        # Apply scalar transform and convert to categorical target
        transformed = scalar_transform(target_scalar)
        target_dist = scalar_to_support(transformed, support_size).to(logits.device)
        # Cross-entropy between predicted logits and target distribution
        return -(target_dist * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def _reward_loss(self, logits: torch.Tensor, target_scalar: torch.Tensor,
                     support_size: int) -> torch.Tensor:
        """Cross-entropy loss for categorical reward prediction."""
        transformed = scalar_transform(target_scalar)
        target_dist = scalar_to_support(transformed, support_size).to(logits.device)
        return -(target_dist * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def _log(self, loss_info: dict, step: int):
        """Log metrics to TensorBoard."""
        for key, value in loss_info.items():
            self.writer.add_scalar(f"loss/{key}", value, step)

    def _save_checkpoint(self, step: int):
        """Save model checkpoint."""
        path = Path("checkpoints") / self.config.game
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": step,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path / f"checkpoint_{step}.pt")

    def _evaluate(self, step: int):
        """Evaluate current model against random player."""
        self.network.eval()
        wins = 0
        losses = 0
        draws = 0

        for _ in range(self.config.eval_games):
            result = self._play_vs_random()
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1

        win_rate = wins / self.config.eval_games
        self.writer.add_scalar("eval/win_rate_vs_random", win_rate, step)
        self.writer.add_scalar("eval/draw_rate_vs_random", draws / self.config.eval_games, step)
        tqdm.write(f"Step {step}: vs random - W:{wins} L:{losses} D:{draws} ({win_rate:.1%} win rate)")

    @torch.no_grad()
    def _play_vs_random(self) -> int:
        """Play one game: model (player 1) vs random (player 2).

        Returns game outcome from model's perspective: 1 (win), -1 (loss), 0 (draw).
        """
        from ..mcts.mcts import MCTS, select_action
        import random

        state = self.game.reset()
        mcts = MCTS(self.network, self.game, self.config, self.device)

        while not state.done:
            legal = self.game.legal_actions(state)
            if state.current_player == 1:
                # Model plays
                obs = self.game.to_tensor(state)
                root = mcts.run(obs, legal, add_noise=False)
                action, _ = select_action(root, temperature=0)
            else:
                # Random plays
                action = random.choice(legal)

            state, _, _ = self.game.step(state, action)

        return state.winner
