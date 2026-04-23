"""MuZero training loop."""

import ctypes
import os
import pickle
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import MuZeroConfig
from ..games.base import Game
from ..model.muzero_net import MuZeroNetwork
from ..model.utils import scalar_transform, scalar_to_support, support_to_scalar, inverse_scalar_transform
from .replay_buffer import ReplayBuffer, _iter_shard_games
from .self_play import run_self_play


def _negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """SimSiam loss: -cos(p, z) per sample. Target z already detached by caller.

    F.normalize eps handles zero-norm vectors safely (they map to zero output,
    giving 0 similarity — which paired with a zero mask contributes no loss).
    """
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=-1)


class MuZeroTrainer:
    """Main training loop for MuZero."""

    def __init__(
        self,
        config: MuZeroConfig,
        game: Game,
        network: MuZeroNetwork,
        run_id: str,
        device: str = "cpu",
        log_dir: str = "runs",
        checkpoints_dir: str = "checkpoints",
    ):
        self.config = config
        self.game = game
        self.network = network.to(device)
        self.device = device
        self.run_id = run_id
        self.checkpoint_dir = Path(checkpoints_dir) / config.game / run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # LR scheduling: piecewise constant decay at milestone fractions of training_steps
        milestones = [int(m * config.training_steps) for m in config.lr_decay_milestones]
        self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=config.lr_decay_factor)

        self.scaler = GradScaler("cuda", enabled=config.use_amp)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, config.game, run_id))
        self.global_step = 0

        # Stockfish injection state. Populated by scripts/train.py via
        # set_injection_shards(paths) when --stockfish-injection-path is passed.
        # _injection_loaded counts games consumed so far from the pool — used as
        # the cursor so resume can pick up where the previous run left off.
        self._injection_shards: list[Path] = []
        self._injection_loaded: int = 0
        # In-memory cache of the currently-open shard's games; drained one game
        # at a time by _inject_stockfish_games to avoid re-reading each shard.
        self._injection_shard_idx: int = 0
        self._injection_shard_games: list = []

    def train(self):
        """Run the full training loop."""
        print(f"Starting MuZero training for {self.config.game}")
        print(f"Device: {self.device}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")

        # Cold-start bootstrap. If a Stockfish pool is attached, bulk-inject
        # enough games to start training on non-trivial buffer diversity.
        # Otherwise fall back to a random self-play round.
        if len(self.replay_buffer) == 0:
            if self._injection_shards:
                n = max(self.config.stockfish_injection_games, self.config.min_buffer_size)
                print(f"Bootstrapping buffer with {n} Stockfish games from injection pool")
                self._inject_stockfish_games(n)
            else:
                print("Generating initial self-play games...")
                self._run_self_play()

        start_step = self.global_step
        pbar = tqdm(range(start_step, self.config.training_steps), desc="Training",
                    initial=start_step, total=self.config.training_steps)
        for step in pbar:
            self.global_step = step

            # Stockfish injection. Runs on every ``interval`` until the pool is
            # exhausted (at which point ``_inject_stockfish_games`` clears
            # ``_injection_shards`` and this branch goes dormant).
            if (
                self._injection_shards
                and self.config.stockfish_injection_interval > 0
                and step > 0
                and step % self.config.stockfish_injection_interval == 0
            ):
                self._inject_stockfish_games(self.config.stockfish_injection_games)

            # Self-play and reanalyze are gated off while the pool is alive.
            # When it exhausts, both flip on automatically for the rest of training.
            pool_alive = bool(self._injection_shards)

            if (
                step > 0
                and step % self.config.self_play_interval == 0
                and not pool_alive
            ):
                self._run_self_play()

            if (
                self.config.reanalyze_interval > 0
                and step % self.config.reanalyze_interval == 0
                and len(self.replay_buffer) >= self.config.min_buffer_size
                and not pool_alive
            ):
                self._reanalyze()

            loss_info = self._train_step()
            self.scheduler.step()

            if step % self.config.log_interval == 0:
                self._log(loss_info, step)
                pbar.set_postfix({
                    "loss": f"{loss_info['total_loss']:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    "buf": len(self.replay_buffer),
                })

            if step > 0 and step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(step)

            if step > 0 and step % self.config.eval_interval == 0:
                self._evaluate(step)

        self._save_checkpoint(self.config.training_steps)
        self.writer.close()
        print("Training complete!")

    def _run_self_play(self):
        """Generate self-play games and add to replay buffer."""
        proc = psutil.Process()
        rss_before = proc.memory_info().rss / 1e9
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            gpu_before = torch.cuda.memory_allocated() / 1e9

        games = run_self_play(
            self.network, self.game, self.config,
            self.config.num_self_play_games, self.device,
            training_step=self.global_step,
        )
        for g in games:
            self.replay_buffer.save_game(g)

        rss_after = proc.memory_info().rss / 1e9
        rss_delta = rss_after - rss_before
        gpu_after = gpu_peak = 0.0
        if self.device == "cuda":
            gpu_after = torch.cuda.memory_allocated() / 1e9
            gpu_peak = torch.cuda.max_memory_allocated() / 1e9

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
        self.writer.add_scalar("memory/rss_gb_before_selfplay", rss_before, self.global_step)
        self.writer.add_scalar("memory/rss_gb_after_selfplay", rss_after, self.global_step)
        self.writer.add_scalar("memory/rss_delta_gb_selfplay", rss_delta, self.global_step)
        if self.device == "cuda":
            self.writer.add_scalar("memory/gpu_alloc_gb_after_selfplay", gpu_after, self.global_step)
            self.writer.add_scalar("memory/gpu_peak_gb_selfplay", gpu_peak, self.global_step)

        gpu_msg = f" | GPU alloc {gpu_after:.2f} GB (peak {gpu_peak:.2f})" if self.device == "cuda" else ""
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        tqdm.write(
            f"Step {self.global_step}: self-play done | "
            f"RSS {rss_before:.2f} → {rss_after:.2f} GB (Δ {rss_delta:+.2f}){gpu_msg}"
        )

    def set_injection_shards(self, shard_paths: list) -> None:
        """Attach a pool of Stockfish shards for in-loop injection.

        Paths are consumed in list order; _inject_stockfish_games walks them via
        self._injection_loaded (cumulative games consumed across shards), so a
        crash mid-window and resume continues from the next un-consumed game.
        For stable resume behaviour, pass paths in a deterministic order
        (e.g. sorted) and don't reshuffle between runs.
        """
        self._injection_shards = [Path(p) for p in shard_paths]

    def _advance_injection_shard(self) -> bool:
        """Load the next shard into ``_injection_shard_games``. Return False if pool exhausted.

        Handles both legacy ``list[GameHistory]`` shards and compact v2 shards
        via ``_iter_shard_games``.
        """
        if self._injection_shard_idx >= len(self._injection_shards):
            return False
        path = self._injection_shards[self._injection_shard_idx]
        self._injection_shard_games = list(_iter_shard_games(path, self.game))
        self._injection_shard_idx += 1
        return True

    def _fastforward_injection(self) -> None:
        """Skip through shards until we reach ``_injection_loaded`` cumulative games.

        Called lazily on first injection after resume: the checkpoint persisted
        the cursor count but not the in-memory shard state, so on cold start we
        need to discard already-consumed games. Loads and throws away whole
        shards; the final partial shard is retained in ``_injection_shard_games``.
        """
        skipped = 0
        target = self._injection_loaded
        while skipped < target:
            if not self._advance_injection_shard():
                # Cursor exceeds pool — shard list shrank between runs. Give up.
                self._injection_shards = []
                return
            shard_size = len(self._injection_shard_games)
            if skipped + shard_size <= target:
                skipped += shard_size
                self._injection_shard_games = []
            else:
                drop = target - skipped
                self._injection_shard_games = self._injection_shard_games[drop:]
                return

    def _inject_stockfish_games(self, n: int) -> None:
        """Inject up to ``n`` next-un-consumed Stockfish games into the buffer.

        Streams shards lazily: loads one shard at a time into
        ``_injection_shard_games``, drains it, advances to the next. When the
        pool is exhausted, logs once and stops injecting for the remainder of
        training (no wrap — the point is a bounded teacher signal).
        """
        if not self._injection_shards:
            return

        # Lazy fast-forward on first call after resume.
        if (
            self._injection_loaded > 0
            and self._injection_shard_idx == 0
            and not self._injection_shard_games
        ):
            self._fastforward_injection()
            if not self._injection_shards:
                return

        inserted = 0
        while inserted < n:
            if not self._injection_shard_games:
                if not self._advance_injection_shard():
                    tqdm.write(
                        f"Step {self.global_step}: Stockfish injection pool exhausted "
                        f"after {self._injection_loaded} games; injection disabled for "
                        f"remainder of window."
                    )
                    self._injection_shards = []
                    break
            g = self._injection_shard_games.pop(0)
            self.replay_buffer.save_game(g)
            inserted += 1
            self._injection_loaded += 1

        self.writer.add_scalar(
            "injection/games_injected_total", self._injection_loaded, self.global_step
        )
        self.writer.add_scalar(
            "injection/buffer_size", len(self.replay_buffer), self.global_step
        )
        tqdm.write(
            f"Step {self.global_step}: injected {inserted} Stockfish games "
            f"(pool {self._injection_loaded} consumed, buffer {len(self.replay_buffer)})"
        )

    def _train_step(self) -> dict:
        """Single training step on a batch from the replay buffer."""
        self.network.train()

        # Beta anneals from per_beta_init → 1.0 over training
        beta = self.config.per_beta_init + (1.0 - self.config.per_beta_init) * (
            self.global_step / max(1, self.config.training_steps)
        )

        batch, game_indices, is_weights = self.replay_buffer.sample_batch(
            self.config.batch_size,
            self.config.num_unroll_steps,
            self.config.td_steps,
            self.config.discount,
            self.game.action_space_size,
            alpha=self.config.per_alpha,
            beta=beta,
        )

        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        target_values = batch["target_values"].to(self.device)
        target_rewards = batch["target_rewards"].to(self.device)
        target_policies = batch["target_policies"].to(self.device)
        is_weights_t = torch.tensor(is_weights, device=self.device)
        use_consistency = bool(getattr(self.config, "use_consistency_loss", False))
        if use_consistency:
            target_observations = batch["target_observations"].to(self.device)  # (B, K+1, C, H, W)
            target_obs_mask = batch["target_obs_mask"].to(self.device)          # (B, K+1)

        # Under Plain Gumbel MuZero the target is π' (softmax over π_net + σ(completedQ));
        # KL matches the paper's L_policy and reports 0 when π_net ≡ π'. Otherwise use
        # cross-entropy on raw visit counts (gradient is identical to KL up to entropy-of-target).
        policy_loss_fn = (
            self._policy_loss_kl if bool(getattr(self.config, "use_gumbel", False))
            else self._policy_loss
        )

        # Loss-weighting mode. See config.use_root_heavy_loss docstring.
        K = self.config.num_unroll_steps
        use_root_heavy = bool(getattr(self.config, "use_root_heavy_loss", False))
        # Per-unroll-step multiplier applied inside the loop. Root (k=0) always at 1.
        # - root-heavy: unroll weight 1/K, no outer scale (matches paper pseudocode)
        # - uniform:    unroll weight 1,   outer scale 1/(K+1) (current default)
        unroll_scale = (1.0 / K if K > 0 else 0.0) if use_root_heavy else 1.0
        outer_scale = 1.0 if use_root_heavy else 1.0 / (K + 1)

        with autocast(device_type=self.device.split(":")[0], enabled=self.config.use_amp):
            hidden, policy_logits, value_logits = self.network.initial_inference_logits(obs)
            value_logits_k0 = value_logits  # save for priority update

            # Per-sample losses at root (k=0) — always full weight
            policy_loss = policy_loss_fn(policy_logits, target_policies[:, 0])
            value_loss = self._value_loss(value_logits, target_values[:, 0],
                                          self.network.value_support_size)
            reward_loss = torch.zeros(obs.shape[0], device=self.device)
            consistency_loss = torch.zeros(obs.shape[0], device=self.device)

            for k in range(self.config.num_unroll_steps):
                hidden, reward_logits, policy_logits, value_logits = \
                    self.network.recurrent_inference_logits(hidden, actions[:, k])

                hidden.register_hook(lambda grad: grad * 0.5)

                policy_loss = policy_loss + unroll_scale * policy_loss_fn(policy_logits, target_policies[:, k + 1])
                value_loss = value_loss + unroll_scale * self._value_loss(value_logits, target_values[:, k + 1],
                                                                          self.network.value_support_size)
                reward_loss = reward_loss + unroll_scale * self._reward_loss(reward_logits, target_rewards[:, k + 1],
                                                                             self.network.reward_support_size)

                if use_consistency:
                    # SimSiam: online branch via dynamics (grad); target branch via representation
                    # of the actual future observation (stop-grad). Matches LightZero
                    # lzero/policy/efficientzero.py consistency loop.
                    dyn_proj = self.network.project(hidden, with_grad=True)
                    with torch.no_grad():
                        target_hidden = self.network.representation(target_observations[:, k + 1])
                        tgt_proj = self.network.project(target_hidden, with_grad=False)
                    consistency_loss = consistency_loss + unroll_scale * (
                        _negative_cosine_similarity(dyn_proj, tgt_proj)
                        * target_obs_mask[:, k + 1]
                    )

            per_sample_loss = outer_scale * (
                policy_loss
                + self.config.value_loss_weight * value_loss
                + reward_loss
                + self.config.consistency_loss_weight * consistency_loss
            )
            total_loss = (is_weights_t * per_sample_loss).mean()

        # Skip optimizer and priority updates on non-finite loss so NaN/Inf
        # doesn't propagate into weights or poison the priority buffer.
        if not torch.isfinite(total_loss):
            tqdm.write(f"Step {self.global_step}: non-finite loss ({total_loss.item()}), skipping step")
            self.optimizer.zero_grad()
            return {
                "total_loss": float("nan"),
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "reward_loss": float("nan"),
                "consistency_loss": float("nan"),
                "grad_norm": float("nan"),
                "amp_scale": float(self.scaler.get_scale()),
                "value_mae": float("nan"),
                "value_target_std": float("nan"),
            }

        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        # clip_grad_norm_ returns the pre-clip total norm — useful as a divergence
        # signal (sudden spike → bad gradient → likely source of loss spikes).
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update PER priorities from root value TD errors. TD error is computed in
        # transformed space (matches target_dist construction). MAE / target_std are
        # reported in raw scalar space so they compare against a meaningful floor:
        # if value_mae ≳ value_target_std, the model is just predicting the batch
        # mean. CE loss hits a floor once targets are concentrated, so MAE-vs-std
        # is the more honest "is the value head learning?" signal.
        with torch.no_grad():
            pred_v_trans = support_to_scalar(value_logits_k0.detach().float(), self.network.value_support_size).squeeze(-1)
            true_v_scalar = target_values[:, 0]
            if self.config.use_scalar_transform:
                true_v_trans = scalar_transform(true_v_scalar)
                pred_v_scalar = inverse_scalar_transform(pred_v_trans)
            else:
                scale = self.config.value_target_scale
                true_v_trans = true_v_scalar * scale
                pred_v_scalar = pred_v_trans / scale if scale != 1.0 else pred_v_trans
            td_errors = (pred_v_trans - true_v_trans).abs().cpu().numpy()
            value_mae = (pred_v_scalar - true_v_scalar).abs().mean().item()
            value_target_std = true_v_scalar.float().std(unbiased=False).item()
        self.replay_buffer.update_priorities(game_indices, td_errors, self.config.per_epsilon)

        return {
            "total_loss": total_loss.item(),
            "policy_loss": (outer_scale * policy_loss.mean()).item(),
            "value_loss": (outer_scale * self.config.value_loss_weight * value_loss.mean()).item(),
            "reward_loss": (outer_scale * reward_loss.mean()).item(),
            "consistency_loss": (outer_scale * self.config.consistency_loss_weight * consistency_loss.mean()).item(),
            "grad_norm": float(grad_norm),
            "amp_scale": float(self.scaler.get_scale()),
            "value_mae": value_mae,
            "value_target_std": value_target_std,
        }

    @torch.no_grad()
    def _reanalyze(self):
        """Re-run MCTS on stored positions with the current network to freshen targets.

        Samples games uniformly (not via PER) to avoid double-biasing: PER already
        concentrates training on high-error positions; applying it here too would
        over-concentrate reanalyze on the same games (muzero-general force_uniform=True).

        Updates both policies and root_values in-place so that make_target's TD
        bootstrap (which reads root_values) also benefits from the fresher estimates.
        No Dirichlet noise: we want clean policy estimates, not exploratory ones
        (consistent with lightzero's reanalyze_noise=False default).
        """
        from ..mcts.mcts import BatchedMCTS, select_action, select_action_gumbel

        use_gumbel = bool(getattr(self.config, "use_gumbel", False))

        _, games = self.replay_buffer.sample_games_for_reanalyze(
            self.config.reanalyze_batch_size
        )
        if not games:
            # Happens early in warmstart runs when buffer is all external_values games.
            return

        # Flatten all non-terminal positions across sampled games into one list.
        # Each entry: (obs tensor, legal_actions list, game ref, position index).
        items = []
        for game in games:
            for pos in range(len(game.policies)):
                if pos < len(game.legal_actions_list):
                    items.append((game.observations[pos], game.legal_actions_list[pos], game, pos))

        if not items:
            return

        self.network.eval()
        batched_mcts = BatchedMCTS(self.network, self.game, self.config, self.device)
        chunk = max(1, getattr(self.config, "num_parallel_games", 1))
        action_space_size = self.game.action_space_size

        positions_updated = 0
        num_games = len(games)
        tqdm.write(f"Step {self.global_step}: reanalyzing {num_games} games ({len(items)} positions)...")
        for start in range(0, len(items), chunk):
            batch = items[start : start + chunk]
            obs_list = [x[0] for x in batch]
            legal_list = [x[1] for x in batch]

            roots = batched_mcts.run_batch(obs_list, legal_list, add_noise=False)

            for (_, _, game, pos), root in zip(batch, roots):
                if float(root.child_visits.sum()) > 0:
                    if use_gumbel:
                        # Plain Gumbel MuZero: π' = softmax(logits + σ(completedQ)) over legals.
                        _, action_probs = select_action_gumbel(
                            root, self.config, action_space_size
                        )
                        game.policies[pos] = action_probs
                    else:
                        # temperature=1.0 → raw visit-count normalization. Under
                        # Sampled MuZero the search prior was renormalized over σ,
                        # so raw visits already approximate Î_β π (Hubert 2021
                        # Proposed Modification).
                        _, action_probs = select_action(root, temperature=1.0)
                        # Pad to full action space size.
                        new_policy = np.zeros(action_space_size, dtype=np.float32)
                        new_policy[: len(action_probs)] = action_probs
                        game.policies[pos] = new_policy
                game.root_values[pos] = root.value
                positions_updated += 1

        tqdm.write(f"Step {self.global_step}: reanalyze done ({positions_updated} positions updated)")
        self.writer.add_scalar("reanalyze/positions_updated", positions_updated, self.global_step)

    def _policy_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-sample cross-entropy between predicted policy and MCTS visit distribution."""
        return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)

    def _policy_loss_kl(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-sample KL(π'_target || π_net) — paper Eq. 12.

        Gradient matches cross-entropy up to H(target) (constant in network params),
        but the reported loss value is more interpretable (→ 0 as π_net → π').
        F.kl_div handles ``target == 0`` safely via the 0·log(0) = 0 convention.
        """
        log_p = F.log_softmax(logits, dim=1)
        return F.kl_div(log_p, targets, reduction="none").sum(dim=1)

    def _value_loss(self, logits: torch.Tensor, target_scalar: torch.Tensor,
                    support_size: int) -> torch.Tensor:
        """Per-sample cross-entropy for categorical value prediction."""
        if self.config.use_scalar_transform:
            transformed = scalar_transform(target_scalar)
        else:
            transformed = target_scalar * self.config.value_target_scale
        target_dist = scalar_to_support(transformed, support_size).to(logits.device)
        return -(target_dist * F.log_softmax(logits, dim=1)).sum(dim=1)

    def _reward_loss(self, logits: torch.Tensor, target_scalar: torch.Tensor,
                     support_size: int) -> torch.Tensor:
        """Per-sample cross-entropy for categorical reward prediction."""
        transformed = scalar_transform(target_scalar) if self.config.use_scalar_transform else target_scalar
        target_dist = scalar_to_support(transformed, support_size).to(logits.device)
        return -(target_dist * F.log_softmax(logits, dim=1)).sum(dim=1)

    def _log(self, loss_info: dict, step: int):
        # Route loss/grad/scale scalars to their own namespaces so TensorBoard
        # groups them sensibly.
        value_keys = {"value_mae", "value_target_std"}
        for key, value in loss_info.items():
            if key in ("grad_norm", "amp_scale"):
                self.writer.add_scalar(f"train/{key}", value, step)
            elif key in value_keys:
                self.writer.add_scalar(f"value/{key[len('value_'):]}", value, step)
            else:
                self.writer.add_scalar(f"loss/{key}", value, step)
        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], step)
        self.writer.add_scalar("train/per_beta", self.config.per_beta_init + (1.0 - self.config.per_beta_init) * (
            step / max(1, self.config.training_steps)
        ), step)

    def _save_checkpoint(self, step: int):
        torch.save({
            "step": step,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "injection_loaded": self._injection_loaded,
        }, self.checkpoint_dir / f"checkpoint_{step}.pt")

        # Only pickle self-play games. Warmstart games are on disk as Stockfish
        # shards and can be reloaded via --warmstart-path on resume; including
        # them in the .buf would rewrite a multi-GB static dataset every
        # checkpoint, OOM-ing the host. total_games is preserved unchanged.
        buf = self.replay_buffer.buffer
        n_self_play = sum(1 for g in buf if not g.external_values)
        if n_self_play == 0:
            tqdm.write(
                f"Step {step}: skipped buffer pickle "
                f"(no self-play games yet — resume via --warmstart-path)"
            )
        else:
            cap = self.config.max_buf_save_games
            self.replay_buffer.save(
                self.checkpoint_dir / f"checkpoint_{step}.buf",
                filter_fn=lambda g: not g.external_values,
                max_games=cap,
            )
            saved = min(n_self_play, cap) if cap is not None else n_self_play
            cap_note = f" (capped; {n_self_play - saved} older dropped)" if cap is not None and n_self_play > cap else ""
            tqdm.write(
                f"Step {step}: saved buffer "
                f"({saved} self-play games{cap_note}; warmstart excluded — "
                f"re-pass --warmstart-path on resume)"
            )

    def load_checkpoint(self, checkpoint_path: str):
        """Load model, optimizer, scheduler, and replay buffer from a checkpoint."""
        from src.config import MuZeroConfig
        torch.serialization.add_safe_globals([MuZeroConfig])
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["step"]
        # Restore injection cursor; absent in pre-injection checkpoints → 0.
        self._injection_loaded = int(ckpt.get("injection_loaded", 0))

        buf_path = Path(checkpoint_path).with_suffix(".buf")
        if buf_path.exists():
            try:
                self.replay_buffer.load(buf_path, game=self.game)
                print(f"Loaded replay buffer ({len(self.replay_buffer)} games)")
            except (EOFError, pickle.UnpicklingError) as e:
                # A .buf truncated by a crashed/OOM'd save shouldn't prevent resume.
                # Fall back to an empty buffer; --warmstart-path can re-seed it.
                print(f"WARNING: {buf_path.name} is corrupt ({type(e).__name__}: {e}); "
                      f"starting with empty buffer")
                self.replay_buffer.buffer = []
                self.replay_buffer._priorities = []
        else:
            print("No saved buffer found — starting with empty buffer")

        print(f"Resumed from step {ckpt['step']}")

    def _evaluate(self, step: int):
        """Evaluate current model against random and (for tictactoe) minimax."""
        self.network.eval()
        n = self.config.eval_games

        wins, losses, draws = self._play_vs_random_batched(n)

        self.writer.add_scalar("eval/win_rate_vs_random", wins / n, step)
        self.writer.add_scalar("eval/draw_rate_vs_random", draws / n, step)
        self.writer.add_scalar("eval/loss_rate_vs_random", losses / n, step)
        tqdm.write(f"Step {step}: vs random  - W:{wins} L:{losses} D:{draws} ({wins/n:.1%} win rate)")

        from ..games.tictactoe import TicTacToe, MinimaxPlayer
        if isinstance(self.game, TicTacToe):
            minimax = MinimaxPlayer()
            wins, losses, draws = 0, 0, 0
            for _ in range(n):
                result = self._play_vs_minimax(minimax)
                if result == 1:
                    wins += 1
                elif result == -1:
                    losses += 1
                else:
                    draws += 1

            self.writer.add_scalar("eval/draw_rate_vs_minimax", draws / n, step)
            self.writer.add_scalar("eval/loss_rate_vs_minimax", losses / n, step)
            tqdm.write(f"Step {step}: vs minimax - W:{wins} L:{losses} D:{draws} ({draws/n:.1%} draw rate)")

    def _agent_action(self, state):
        """Model's greedy action. Under use_gumbel routes through BatchedMCTS
        (Plain Gumbel MuZero path), otherwise serial MCTS + temperature=0."""
        from ..mcts.mcts import MCTS, BatchedMCTS, select_action, select_action_gumbel

        legal = self.game.legal_actions(state)
        obs = self.game.to_tensor(state)
        use_gumbel = bool(getattr(self.config, "use_gumbel", False))
        if use_gumbel:
            batched = BatchedMCTS(self.network, self.game, self.config, self.device)
            root = batched.run_batch([obs], [legal], add_noise=False)[0]
            action, _ = select_action_gumbel(root, self.config, self.game.action_space_size)
        else:
            mcts = MCTS(self.network, self.game, self.config, self.device)
            root = mcts.run(obs, legal, add_noise=False)
            action, _ = select_action(root, temperature=0)
        return action

    @torch.no_grad()
    def _play_vs_random_batched(self, n_games: int) -> tuple[int, int, int]:
        """Play n_games vs random in parallel; agent on player 1, random on player -1.

        All agent-turn games are collected into one BatchedMCTS.run_batch call per
        ply — scales to ~1 batched forward pass per simulation instead of one per
        game, per move.
        """
        import random
        from ..mcts.mcts import BatchedMCTS, select_action, select_action_gumbel

        use_gumbel = bool(getattr(self.config, "use_gumbel", False))
        batched = BatchedMCTS(self.network, self.game, self.config, self.device)
        states = [self.game.reset() for _ in range(n_games)]

        while any(not s.done for s in states):
            for i, s in enumerate(states):
                if not s.done and s.current_player == -1:
                    a = random.choice(self.game.legal_actions(s))
                    states[i], _, _ = self.game.step(s, a)

            agent_idx = [i for i, s in enumerate(states) if not s.done and s.current_player == 1]
            if not agent_idx:
                continue
            obs_list = [self.game.to_tensor(states[i]) for i in agent_idx]
            legal_list = [self.game.legal_actions(states[i]) for i in agent_idx]
            roots = batched.run_batch(obs_list, legal_list, add_noise=False)
            for k, i in enumerate(agent_idx):
                if use_gumbel:
                    a, _ = select_action_gumbel(roots[k], self.config, self.game.action_space_size)
                else:
                    a, _ = select_action(roots[k], temperature=0)
                states[i], _, _ = self.game.step(states[i], a)

        wins = sum(1 for s in states if s.winner == 1)
        losses = sum(1 for s in states if s.winner == -1)
        draws = sum(1 for s in states if s.winner == 0)
        return wins, losses, draws

    @torch.no_grad()
    def _play_vs_minimax(self, minimax) -> int:
        """Play one game: model (player 1) vs minimax (player 2).

        Minimax plays perfectly — optimal result is a draw. Any loss means
        the agent made a suboptimal move. Wins are impossible against perfect play.
        """
        state = self.game.reset()
        while not state.done:
            if state.current_player == 1:
                action = self._agent_action(state)
            else:
                action = minimax.best_action(state.board, player=-1)
            state, _, _ = self.game.step(state, action)

        return state.winner
