"""MuZero training loop."""

import ctypes
import os
import pickle
import queue
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import MuZeroConfig
from ..games.base import Game
from ..model.muzero_net import MuZeroNetwork
from ..model.utils import scalar_transform, scalar_to_support, support_to_scalar, inverse_scalar_transform
from .replay_buffer import (
    GameHistory, ReplayBuffer, _iter_shard_games, _shard_record_count,
    _sparsify_policy,
)
from .representation_probe import compute_repr_metrics
from .self_play import (run_self_play, get_material_value_weight, get_material_head_weight,
                        get_warmstart_sample_frac, get_tb_policy_weight)


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

        # CUDA inference perf knobs. cudnn.benchmark auto-tunes conv algorithms
        # for the network's static shapes (one-time cost first call). TF32 lets
        # Ampere/Ada use tensor cores even without explicit fp16 autocast.
        # Both are no-ops on CPU; idempotent on repeated calls.
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # torch.compile the MCTS-side inference methods only. MCTS uses the
        # scalar variants (initial_inference / recurrent_inference) under
        # network.eval() with no autograd — this is the high-frequency path
        # (hundreds of forwards per ply during self-play) and the only one
        # where compile is clearly safe.
        #
        # We deliberately DO NOT compile initial_inference_logits /
        # recurrent_inference_logits — those are the trainer's path and run
        # under fp16 autocast WITH autograd active. The autocast +
        # autograd + inductor combo SIGSEGV'd at production-net depth (16
        # blocks) on 2026-05-07 run; eager mode is fine, and training still
        # gets autocast's tensor-core speedup at the op level.
        #
        # Bound-method compile: instance attributes hold the compiled
        # callables; underlying parameters are unchanged so save/load
        # state_dict work without translation. mode='default' = inductor
        # fusion only, no CUDA graphs.
        if (
            getattr(self.config, "compile_network", False)
            and device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            self.network.initial_inference = torch.compile(
                self.network.initial_inference, mode="default"
            )
            self.network.recurrent_inference = torch.compile(
                self.network.recurrent_inference, mode="default"
            )
        self.run_id = run_id
        self.checkpoint_dir = Path(checkpoints_dir) / config.game / run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # LR scheduling: optional linear warmup → piecewise constant decay.
        # Milestones are fractions of training_steps. Warmup, if enabled,
        # ramps lr from lr*start_factor → lr over lr_warmup_steps; the
        # decay scheduler then takes over (milestones are measured in
        # absolute steps, so they fire correctly after warmup).
        milestones = [int(m * config.training_steps) for m in config.lr_decay_milestones]
        decay_scheduler = MultiStepLR(
            self.optimizer, milestones=milestones, gamma=config.lr_decay_factor
        )
        if config.lr_warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-4,          # ramp from tiny → full lr
                end_factor=1.0,
                total_iters=config.lr_warmup_steps,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[config.lr_warmup_steps],
            )
        else:
            self.scheduler = decay_scheduler

        self.scaler = GradScaler("cuda", enabled=config.use_amp)
        self.replay_buffer = ReplayBuffer(
            config.replay_buffer_size,
            warmstart_max_size=getattr(config, "warmstart_buffer_size", None),
            decisive_retention_multiplier=getattr(config, "decisive_retention_multiplier", 1.0),
        )
        # Two-phase buffer sizing: during the supervised warmstart phase there is
        # no self-play to protect the anchor from, so let warmstart games fill the
        # WHOLE buffer (single-pool) for maximum supervised diversity. The
        # training loop trims the warmstart pool back to ``warmstart_buffer_size``
        # and re-enables the two-pool anchor at the self-play boundary. (On resume
        # past the boundary the loop immediately restores the anchor cap below.)
        if getattr(config, "self_play_warmup_steps", 0) > 0:
            self.replay_buffer.warmstart_max_size = None

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, config.game, run_id))
        self.global_step = 0
        # Throughput tracking for train/steps_per_sec (computed across log windows).
        self._last_log_t: float | None = None
        self._last_log_step: int | None = None

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

        # TB endgame ANCHOR injection state (strategy_2026_07_02.md). Unlike the
        # bounded Stockfish pool, the anchor CYCLES its archive forever — a
        # persistent, never-annealed supply of tablebase-optimal demonstration
        # games (the supervised-proxy signal, KQvK 0.91) injected into the
        # rolling self-play pool. Compact dicts stay in RAM; each injection
        # reconstructs a FRESH GameHistory so buffer entries never alias.
        self._tb_anchor_dicts: list[dict] = []
        self._tb_anchor_cursor: int = 0
        self._tb_anchor_injected: int = 0
        tb_anchor_path = getattr(config, "tb_anchor_path", None)
        if tb_anchor_path:
            self._load_tb_anchor(tb_anchor_path)
        # Lazily-built fixed held-out probe set for the in-loop representation
        # informativeness metric (repr/* scalars). See representation_probe.py.
        self._repr_probe_set = None

        # --- Batch prefetch (overlap the ~141ms CPU sample_batch with the GPU
        # step) ---------------------------------------------------------------
        # sample_batch is single-threaded CPU work that idles the GPU each step.
        # A background thread samples batch N+1 while the main thread runs the
        # GPU forward/backward for N; the queue's maxsize provides backpressure
        # so the prefetcher paces itself to consumption. The lock serializes
        # buffer reads (sample_batch) against buffer writes (save_game,
        # update_priorities) — the GPU step holds no lock, so it overlaps fully.
        # Disabled by default; enable via config.prefetch_batches.
        self.prefetch_enabled = bool(getattr(config, "prefetch_batches", False))
        self._buffer_lock = threading.Lock()
        self._prefetch_queue: queue.Queue | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_stop = threading.Event()
        self._prefetch_error: BaseException | None = None

        # Seeded-endgame conversion monitoring (TensorBoard seed/*). Lazily-opened
        # Syzygy handle + per-FEN wdl cache to classify each seed's winning side.
        self._seed_tb = None
        self._seed_wdl_cache: dict[str, int] = {}

    def train(self):
        """Run the full training loop."""
        print(f"Starting MuZero training for {self.config.game}")
        print(f"Device: {self.device}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")

        # Cold-start bootstrap. If a Stockfish pool is attached, bulk-inject
        # enough games to start training on non-trivial buffer diversity.
        # Otherwise fall back to a random self-play round.
        #
        # Resume-with-empty-buffer trap: when resuming from a .pt whose paired
        # .buf is missing AND the injection cursor was already at pool-end
        # (e.g. resuming from checkpoint_30000.pt of a prior run that had
        # consumed all 32k warmstart games), the inject call fills 0 games
        # and the first train_step crashes on np.random.choice(0,...). Guard
        # by falling through to self-play if the inject leaves us empty.
        if len(self.replay_buffer) == 0:
            if self._injection_shards:
                n = max(self.config.stockfish_injection_games, self.config.min_buffer_size)
                print(f"Bootstrapping buffer with {n} Stockfish games from injection pool")
                self._inject_stockfish_games(n)
            if len(self.replay_buffer) == 0:
                print("Buffer still empty after injection bootstrap — "
                      "falling back to self-play to seed the buffer")
                self._run_self_play()

        start_step = self.global_step
        # Buffer is guaranteed non-empty here (bootstrap above), so the prefetch
        # producer can sample immediately. No-op when prefetch is disabled. The
        # thread is a daemon, so an abnormal exit (crash / KeyboardInterrupt) can't
        # wedge process shutdown; _stop_prefetch() below handles normal completion.
        self._start_prefetch()
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

            # TB endgame anchor injection: a cycling, never-exhausting supply of
            # tablebase-optimal demonstration games (the validated supervised-
            # proxy signal) mixed into the rolling buffer at a constant rate.
            tb_anchor_interval = int(getattr(self.config, "tb_anchor_interval", 0))
            if (
                self._tb_anchor_dicts
                and tb_anchor_interval > 0
                and step > 0
                and step % tb_anchor_interval == 0
            ):
                self._inject_tb_anchor_games(
                    int(getattr(self.config, "tb_anchor_games", 0)))

            # Self-play / reanalyze gating. Two modes:
            #   (A) Explicit two-phase: if ``self_play_warmup_steps`` is set,
            #       both are OFF until that step (pure supervised warmstart
            #       pretrain), then ON for the rest of training — regardless of
            #       whether the injection pool still has games.
            #   (legacy) Pool gate: otherwise, both are OFF while the injection
            #       pool is alive and flip on when it exhausts.
            warmup = getattr(self.config, "self_play_warmup_steps", 0)
            if warmup > 0:
                selfplay_on = step >= warmup
                # Phase-dependent buffer cap. Phase 1 (selfplay_on False): no cap
                # → warmstart fills the whole buffer. Phase 2: trim to the anchor
                # size and switch on the two-pool so self-play can't displace it.
                anchor = getattr(self.config, "warmstart_buffer_size", None)
                desired_cap = anchor if selfplay_on else None
                if self.replay_buffer.warmstart_max_size != desired_cap:
                    # Lock: trim_warmstart_to mutates the buffer (shrinks it),
                    # which races the prefetch thread's sample_batch read
                    # (np.random.choice(n) then buffer[g_idx] -> IndexError if the
                    # buffer shrank in between). Same lock sample_batch holds.
                    with self._buffer_lock:
                        if desired_cap is not None:
                            n_trimmed = self.replay_buffer.trim_warmstart_to(desired_cap)
                            tqdm.write(
                                f"Step {step}: warmstart phase complete — trimmed "
                                f"{n_trimmed} warmstart games to anchor={desired_cap}, "
                                f"two-pool enabled, self-play ON."
                            )
                        self.replay_buffer.warmstart_max_size = desired_cap
            else:
                selfplay_on = not bool(self._injection_shards)

            if (
                step > 0
                and step % self.config.self_play_interval == 0
                and selfplay_on
            ):
                self._run_self_play()

            if (
                self.config.reanalyze_interval > 0
                and step % self.config.reanalyze_interval == 0
                and len(self.replay_buffer) >= self.config.min_buffer_size
                and selfplay_on
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

            if (getattr(self.config, "repr_probe_interval", 0) > 0
                    and step % self.config.repr_probe_interval == 0):
                self._log_representation_probe(step)

        self._stop_prefetch()
        self._save_checkpoint(self.config.training_steps)
        self.writer.close()
        print("Training complete!")

    def _memory_snapshot(self, stage: str) -> None:
        """Log granular memory metrics under memory/* tag at current global_step.

        ``stage`` is appended to each tag (e.g. ``memory/rss_gb_after_selfplay``).
        Captures process-level RSS/USS/PSS, torch GPU allocator stats, replay
        buffer logical bytes (with view-parent pinning detection), and optimizer
        state size. Used at self-play boundaries; cheap enough to call
        per-step but currently only fires around self-play.
        """
        proc = psutil.Process()
        try:
            mfi = proc.memory_full_info()
            rss = mfi.rss / 1e9
            uss = getattr(mfi, "uss", 0) / 1e9
            pss = getattr(mfi, "pss", 0) / 1e9
            shared = getattr(mfi, "shared", 0) / 1e9
        except Exception:
            mfi = proc.memory_info()
            rss = mfi.rss / 1e9
            uss = pss = shared = 0.0

        step = self.global_step
        self.writer.add_scalar(f"memory/rss_gb_{stage}", rss, step)
        self.writer.add_scalar(f"memory/uss_gb_{stage}", uss, step)
        self.writer.add_scalar(f"memory/pss_gb_{stage}", pss, step)
        self.writer.add_scalar(f"memory/shared_gb_{stage}", shared, step)

        # GPU allocator detail.
        if self.device == "cuda":
            self.writer.add_scalar(f"memory/gpu_alloc_gb_{stage}",
                                   torch.cuda.memory_allocated() / 1e9, step)
            self.writer.add_scalar(f"memory/gpu_reserved_gb_{stage}",
                                   torch.cuda.memory_reserved() / 1e9, step)
            try:
                stats = torch.cuda.memory_stats()
                host_alloc = stats.get("host_alloc_bytes.all.current", 0) / 1e9
                self.writer.add_scalar(f"memory/gpu_host_alloc_gb_{stage}", host_alloc, step)
            except Exception:
                pass

        # Replay buffer breakdown (cheap — iterates lists, no copies).
        try:
            nb = self.replay_buffer.nbytes()
            self.writer.add_scalar(f"memory/buffer_total_gb_{stage}", nb["total"] / 1e9, step)
            self.writer.add_scalar(f"memory/buffer_obs_gb_{stage}", nb["observations"] / 1e9, step)
            self.writer.add_scalar(f"memory/buffer_policies_gb_{stage}", nb["policies"] / 1e9, step)
            self.writer.add_scalar(f"memory/buffer_view_parents_gb_{stage}",
                                   nb["view_parents"] / 1e9, step)
        except Exception:
            pass

        # Optimizer state (Adam: 2 moments per param).
        try:
            opt_bytes = sum(
                t.numel() * t.element_size()
                for s in self.optimizer.state.values()
                for t in s.values() if torch.is_tensor(t)
            )
            self.writer.add_scalar(f"memory/optimizer_state_gb_{stage}", opt_bytes / 1e9, step)
        except Exception:
            pass

    def _run_self_play(self):
        """Generate self-play games and add to replay buffer."""
        proc = psutil.Process()
        rss_before = proc.memory_info().rss / 1e9
        if self.device == "cuda":
            # Release fragmented allocator reservations before self-play.
            # Training-step autograd + reanalyze leave the CUDA caching
            # allocator with high reserved-vs-active ratios; without an
            # empty_cache(), self-play's tree allocations may force
            # cudaFree-cycling syncs (observed: 3× per-ply slowdown after
            # the first reanalyze co-fire on 2026-05-06 run). One call
            # before self-play resets the reservation pool cheaply (the
            # cache is rebuilt from new allocations as needed).
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gpu_before = torch.cuda.memory_allocated() / 1e9

        self._memory_snapshot("before_selfplay")

        sp_t0 = time.monotonic()
        games = run_self_play(
            self.network, self.game, self.config,
            self.config.num_self_play_games, self.device,
            training_step=self.global_step,
        )
        sp_secs = time.monotonic() - sp_t0
        with self._buffer_lock:
            for g in games:
                self.replay_buffer.save_game(g)

        rss_after = proc.memory_info().rss / 1e9
        rss_delta = rss_after - rss_before
        gpu_after = gpu_peak = 0.0
        if self.device == "cuda":
            gpu_after = torch.cuda.memory_allocated() / 1e9
            gpu_peak = torch.cuda.max_memory_allocated() / 1e9

        n_games = len(games)
        avg_length = sum(len(g) - 1 for g in games) / n_games
        outcomes = [g.game_outcome for g in games]
        p1_wins = sum(1 for o in outcomes if o == 1)
        p2_wins = sum(1 for o in outcomes if o == -1)
        draws = sum(1 for o in outcomes if o == 0)

        # Draw-type breakdown (chess only), by replaying each drawn game through
        # python-chess and inspecting the final position:
        #   threefold     — is_repetition(3)
        #   no_progress   — halfmove_clock >= 100 (50/75-move)
        #   computer      — stalemate OR insufficient material (natural engine draw)
        #   plycap        — none of the above: game hit the ply cap (or replay-fail)
        threefold_draws = no_progress_draws = computer_draws = plycap_draws = 0
        if getattr(self.config, "game", None) == "chess" and draws > 0:
            import chess as _chess
            from ..games.chess import _action_to_move as _a2m
            for g in games:
                if g.game_outcome != 0:
                    continue
                # Replay from the game's OWN start (seeded endgame games begin from a
                # start_fen, not the standard position) — else every move is illegal
                # from move 1 and the game falls into the plycap residual, mislabeling
                # all seeded draws as ply-cap when they actually end by 3fold/insufficient.
                b = _chess.Board(getattr(g, "start_fen", None) or _chess.STARTING_FEN); ok = True
                try:
                    for a in g.actions:
                        mv = _a2m(int(a), b)
                        if mv is None or mv not in b.legal_moves:
                            ok = False; break
                        b.push(mv)
                except Exception:
                    ok = False
                if ok and b.is_repetition(3):
                    threefold_draws += 1
                elif ok and b.halfmove_clock >= 100:
                    no_progress_draws += 1
                elif ok and (b.is_stalemate() or b.is_insufficient_material()):
                    computer_draws += 1
                else:
                    plycap_draws += 1              # ply-cap / forced-draw / replay-fail residual

        total_plies = sum(len(g) - 1 for g in games)
        self.writer.add_scalar("self_play/avg_game_length", avg_length, self.global_step)
        self.writer.add_scalar("self_play/p1_win_rate", p1_wins / n_games, self.global_step)
        self.writer.add_scalar("self_play/p2_win_rate", p2_wins / n_games, self.global_step)
        self.writer.add_scalar("self_play/draw_rate", draws / n_games, self.global_step)
        # Draw-type breakdown (these four + p1_win + p2_win sum to 1.0).
        self.writer.add_scalar("self_play/draw_threefold_rate", threefold_draws / n_games, self.global_step)
        self.writer.add_scalar("self_play/draw_no_progress_rate", no_progress_draws / n_games, self.global_step)
        self.writer.add_scalar("self_play/draw_computer_rate", computer_draws / n_games, self.global_step)
        self.writer.add_scalar("self_play/draw_plycap_rate", plycap_draws / n_games, self.global_step)
        # Resignation as its own OUTCOME CATEGORY. Resigned games are relabeled
        # decisive (game_outcome = ±1), so they're already folded into
        # p1/p2_win_rate above. Here we split them out so the three top-level
        # categories are mutually exclusive and sum to 1.0:
        #     win_natural_rate + resignation_rate + draw_rate == 1.0
        # (draws are never resigned). resign_p{1,2}_win_rate sum to
        # resignation_rate, mirroring the draw-type sub-breakdown. 0 throughout
        # when resign_enabled=False.
        resigned_games = sum(1 for g in games if getattr(g, "resigned", False))
        resign_p1 = sum(1 for g in games if getattr(g, "resigned", False) and g.game_outcome == 1)
        resign_p2 = sum(1 for g in games if getattr(g, "resigned", False) and g.game_outcome == -1)
        natural_decisive = (p1_wins + p2_wins) - resigned_games
        self.writer.add_scalar("self_play/resignation_rate", resigned_games / n_games, self.global_step)
        self.writer.add_scalar("self_play/resign_p1_win_rate", resign_p1 / n_games, self.global_step)
        self.writer.add_scalar("self_play/resign_p2_win_rate", resign_p2 / n_games, self.global_step)
        self.writer.add_scalar("self_play/win_natural_rate", natural_decisive / n_games, self.global_step)
        # Resignation calibration (AlphaZero-style holdout): games that hit the
        # resign trigger but were played to completion, and of those the fraction
        # where the would-have-resigned side did NOT actually lose (false
        # positive). If resign_false_positive_rate exceeds ~5%, resign_threshold
        # is too aggressive (it's truncating winnable/drawable games as losses).
        holdout_triggered = sum(1 for g in games if getattr(g, "resign_holdout", False))
        resign_fp = sum(1 for g in games if getattr(g, "resign_false_positive", False))
        self.writer.add_scalar("self_play/resign_holdout_rate", holdout_triggered / n_games, self.global_step)
        self.writer.add_scalar("self_play/resign_false_positive_rate",
                               resign_fp / max(1, holdout_triggered), self.global_step)
        # TB rollout fill: fraction of games truncated at a decisive in-TB ply and
        # finished with TB-optimal play (won-but-unconverted adjudications).
        if getattr(self.config, "tb_rollout_fill", False):
            filled = sum(1 for g in games if getattr(g, "tb_filled", False))
            self.writer.add_scalar(
                "self_play/tb_fill_rate", filled / n_games, self.global_step)
        # Root tablebase reach rate: fraction of games that ever simplified into a
        # ≤tb_max_pieces position (where root TB probing can fire). Tells us how
        # often the model even reaches the endgame stage the probe targets.
        if getattr(self.config, "tb_root_probe", False):
            tb_reached = sum(1 for g in games if getattr(g, "reached_tb", False))
            self.writer.add_scalar("self_play/tb_reach_rate", tb_reached / n_games, self.global_step)
        # Seeded-endgame conversion (seed/* scalars) — no-op unless endgame seeding
        # is on (games carry start_fen). Measures how often the model converts the
        # tablebase-winning seeds it was started from, split by winning color.
        self._log_seed_conversion(games)
        self.writer.add_scalar("self_play/buffer_size", len(self.replay_buffer), self.global_step)
        # Buffer composition: fraction of self-play (non-warmstart) games in the
        # buffer that are decisive — the quantity decisive_retention_multiplier is
        # meant to lift (vs the ~self-play draw rate it would otherwise track).
        _sp = [g for g in self.replay_buffer.buffer if not g.external_values]
        if _sp:
            self.writer.add_scalar(
                "self_play/buffer_decisive_frac",
                sum(1 for g in _sp if g.game_outcome != 0.0) / len(_sp),
                self.global_step,
            )
        # Throughput — self-play is the wall-clock bottleneck; track it explicitly.
        self.writer.add_scalar("self_play/seconds", sp_secs, self.global_step)
        if sp_secs > 0:
            self.writer.add_scalar("self_play/games_per_sec", n_games / sp_secs, self.global_step)
            self.writer.add_scalar("self_play/plies_per_sec", total_plies / sp_secs, self.global_step)
        # Console summary of the self-play batch — outcome breakdown is the key
        # draw-basin health signal; surface it in the log, not just TensorBoard.
        tqdm.write(
            f"Step {self.global_step}: self-play batch — "
            f"Wwin={p1_wins / n_games:.2f} Bwin={p2_wins / n_games:.2f} "
            f"draw={draws / n_games:.2f} "
            f"[3fold={threefold_draws / n_games:.2f} 50mv={no_progress_draws / n_games:.2f} "
            f"comp={computer_draws / n_games:.2f} plycap={plycap_draws / n_games:.2f}], "
            f"avg_len={avg_length:.0f}, {n_games} games in {sp_secs:.0f}s"
        )
        self.writer.add_scalar("memory/rss_gb_before_selfplay", rss_before, self.global_step)
        self.writer.add_scalar("memory/rss_gb_after_selfplay", rss_after, self.global_step)
        self.writer.add_scalar("memory/rss_delta_gb_selfplay", rss_delta, self.global_step)
        if self.device == "cuda":
            self.writer.add_scalar("memory/gpu_alloc_gb_after_selfplay", gpu_after, self.global_step)
            self.writer.add_scalar("memory/gpu_peak_gb_selfplay", gpu_peak, self.global_step)

        # Granular post-selfplay snapshot — buffer detail, USS/PSS, GPU host alloc.
        self._memory_snapshot("after_selfplay")

        gpu_msg = f" | GPU alloc {gpu_after:.2f} GB (peak {gpu_peak:.2f})" if self.device == "cuda" else ""
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        tqdm.write(
            f"Step {self.global_step}: self-play done | "
            f"RSS {rss_before:.2f} → {rss_after:.2f} GB (Δ {rss_delta:+.2f}){gpu_msg}"
        )

    def _log_seed_conversion(self, games):
        """Log seeded-endgame conversion to TensorBoard (``seed/*``).

        For games started from a tablebase seed (``start_fen`` set), classify the
        seed's winning side via Syzygy WDL and measure how often the model actually
        won it — split by winning COLOR, which both tracks conversion AND verifies
        the model trains on both white- and black-advantage endgames.

        ``game_outcome`` is white-POV (+1 white win / -1 black win / 0 draw). A
        decisive outcome with ``resigned``False is a real checkmate (the only
        natural decisive chess result; stalemate/threefold/50-move are draws). So:
          conversion_rate = decisive wins for the winning side / won seeds (incl. resignation)
          mate_rate       = real checkmates / won seeds (matches the offline monitor)
          draw_rate       = drawn / won seeds (the failure mode)
          loss_rate       = winning side actually lost / won seeds (blundered the win)
        No-op (returns) unless seeding is on. Cheap: one cached WDL probe per seed."""
        seeded = [g for g in games if getattr(g, "start_fen", None)]
        if not seeded:
            return
        import chess
        import chess.syzygy
        if self._seed_tb is None:
            tb_path = getattr(self.config, "tb_path", None) or "data/syzygy"
            try:
                self._seed_tb = chess.syzygy.open_tablebase(tb_path)
            except Exception:
                self._seed_tb = False
        if not self._seed_tb:
            return

        won = {"white": 0, "black": 0}
        conv = {"white": 0, "black": 0}
        mate = {"white": 0, "black": 0}
        drawn = {"white": 0, "black": 0}
        lost = {"white": 0, "black": 0}
        draw_seed = draw_seed_held = 0
        for g in seeded:
            fen = g.start_fen
            board = chess.Board(fen)
            wdl = self._seed_wdl_cache.get(fen)
            if wdl is None:
                try:
                    wdl = int(self._seed_tb.probe_wdl(board))
                except Exception:
                    wdl = -99
                self._seed_wdl_cache[fen] = wdl
            o = g.game_outcome
            resigned = bool(getattr(g, "resigned", False))
            if wdl == 2:                       # side to move (= winning color) wins
                color = "white" if board.turn == chess.WHITE else "black"
                won[color] += 1
                win_o = 1.0 if board.turn == chess.WHITE else -1.0
                if o == win_o:
                    conv[color] += 1
                    if not resigned:
                        mate[color] += 1
                elif o == 0:
                    drawn[color] += 1
                else:
                    lost[color] += 1
            elif wdl == 0:                     # true draw seed (won/drawn boundary)
                draw_seed += 1
                draw_seed_held += int(o == 0)

        tw, tb_ = won["white"], won["black"]
        total_won = tw + tb_
        s = self.global_step

        def rate(num, den):
            return (num / den) if den else float("nan")

        # Balance verification: counts of white- vs black-winning seeds the model
        # actually played this batch (should track the 50/50 archive).
        self.writer.add_scalar("seed/won_count", total_won, s)
        self.writer.add_scalar("seed/white_winning_count", tw, s)
        self.writer.add_scalar("seed/black_winning_count", tb_, s)
        self.writer.add_scalar("seed/draw_seed_count", draw_seed, s)
        if not total_won:
            return
        c = conv["white"] + conv["black"]
        m = mate["white"] + mate["black"]
        d = drawn["white"] + drawn["black"]
        l = lost["white"] + lost["black"]
        self.writer.add_scalar("seed/conversion_rate", rate(c, total_won), s)   # headline
        self.writer.add_scalar("seed/mate_rate", rate(m, total_won), s)
        self.writer.add_scalar("seed/draw_rate", rate(d, total_won), s)
        self.writer.add_scalar("seed/loss_rate", rate(l, total_won), s)
        self.writer.add_scalar("seed/conversion_white", rate(conv["white"], tw), s)
        self.writer.add_scalar("seed/conversion_black", rate(conv["black"], tb_), s)
        if draw_seed:
            self.writer.add_scalar("seed/draw_seed_hold_rate", rate(draw_seed_held, draw_seed), s)
        tqdm.write(
            f"Step {s}: seed conversion — {c}/{total_won} = {rate(c, total_won):.1%} "
            f"(mate {rate(m, total_won):.1%}, draw {rate(d, total_won):.1%}, "
            f"loss {rate(l, total_won):.1%}) | won W:{tw} B:{tb_} "
            f"conv W:{rate(conv['white'], tw):.0%} B:{rate(conv['black'], tb_):.0%}"
        )

    def _build_repr_probe_set(self):
        """Sample a FIXED held-out set of (stacked_obs, stm_eval, stm_outcome) for
        the in-loop representation probe. Prefers Stockfish-labelled positions
        (injection shards, then warmstart games in the buffer); falls back to
        self-play game outcomes (eval label NaN -> repr/r2_eval is NaN). Built
        once and reused so the repr/* trajectory is comparable across steps.
        Returns (obs[M,C,H,W] f32, eval[M] f32, outcome[M] f32) or None.
        """
        hf = getattr(self.config, "history_frames", 1)
        target = int(getattr(self.config, "repr_probe_positions", 768))
        rng = np.random.default_rng(12345)
        obs_l, ev_l, out_l = [], [], []

        def add_from_games(games, need_ev):
            for gh in games:
                n = len(gh.external_values) if need_ev else len(gh.actions)
                if n <= 0:
                    continue
                for ply in rng.choice(n, size=min(6, n), replace=False):
                    ply = int(ply)
                    obs_l.append(gh._stack_history(ply, hf).numpy())
                    stm_white = (ply % 2 == 0)
                    out_l.append(float(gh.game_outcome) * (1.0 if stm_white else -1.0))
                    ev_l.append(float(gh.external_values[ply]) if need_ev else float("nan"))
                if len(obs_l) >= target:
                    break

        if self._injection_shards:
            for p in self._injection_shards:
                try:
                    add_from_games(list(_iter_shard_games(p, self.game)), need_ev=True)
                except Exception:
                    continue
                if len(obs_l) >= target:
                    break
        if len(obs_l) < target:
            add_from_games([g for g in self.replay_buffer.buffer if g.external_values], need_ev=True)
        if len(obs_l) < target // 2:
            add_from_games([g for g in self.replay_buffer.buffer if not g.external_values], need_ev=False)

        if len(obs_l) < 50:
            return None
        return (np.asarray(obs_l, dtype=np.float32),
                np.asarray(ev_l, dtype=np.float32),
                np.asarray(out_l, dtype=np.float32))

    def _log_representation_probe(self, step: int) -> None:
        """Log representation-informativeness metrics (repr/*) on the fixed probe
        set. r2_eval = linear decodability of Stockfish eval from the frozen
        latent — the leading indicator of draw-basin collapse (falls while value
        LOSS can look healthy). Runs under no_grad on the long repr_probe_interval.
        """
        if self._repr_probe_set is None:
            self._repr_probe_set = self._build_repr_probe_set()
            if self._repr_probe_set is None:
                return
        obs, ev, out = self._repr_probe_set
        H = []
        with torch.no_grad():
            for s in range(0, len(obs), 256):
                x = torch.from_numpy(obs[s:s + 256]).to(self.device)
                h = self.network.representation(x).reshape(x.shape[0], -1).float().cpu().numpy()
                H.append(h)
        m = compute_repr_metrics(np.concatenate(H, 0), ev, out, seed=0)
        self.writer.add_scalar("repr/r2_eval", m["r2_eval"], step)
        self.writer.add_scalar("repr/r2_outcome", m["r2_out"], step)
        self.writer.add_scalar("repr/crosspos_cos", m["crosspos_cos"], step)
        self.writer.add_scalar("repr/eff_rank", m["eff_rank"], step)
        self.writer.add_scalar("repr/sign_acc", m["sign_acc"], step)
        tqdm.write(f"Step {step}: repr probe — r2_eval={m['r2_eval']:.3f} "
                   f"r2_out={m['r2_out']:.3f} cpc={m['crosspos_cos']:.3f} "
                   f"eff_rank={m['eff_rank']:.1f} (n={m['n']})")

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
        # Skip whole already-consumed shards CHEAPLY via their header record-count
        # (no per-game reconstruction). Only the final partial shard is loaded and
        # reconstructed. Previously this loaded+reconstructed every skipped shard —
        # replaying ~19k games' moves through the CPU engine on resume, which
        # looked like a multi-minute hang (GPU idle, CPU pegged).
        skipped = 0
        target = self._injection_loaded
        while skipped < target:
            if self._injection_shard_idx >= len(self._injection_shards):
                # Cursor exceeds pool — shard list shrank between runs. Give up.
                self._injection_shards = []
                return
            path = self._injection_shards[self._injection_shard_idx]
            count = _shard_record_count(path)
            if skipped + count <= target:
                # Whole shard already consumed — advance index, no reconstruction.
                skipped += count
                self._injection_shard_idx += 1
            else:
                # Partial shard straddles the cursor: reconstruct THIS one (advances
                # the index) and drop the already-consumed prefix.
                self._advance_injection_shard()
                drop = target - skipped
                self._injection_shard_games = self._injection_shard_games[drop:]
                return

    def _load_tb_anchor(self, path: str) -> None:
        """Load the TB anchor archive (v2 compact shards) as raw compact dicts.

        Dicts (not GameHistories) are cached: observations are rebuilt per
        injection via ``from_compact_dict`` so repeated injections of the same
        game never share mutable state (birth stamps, reanalyze, priorities).
        """
        import random as _random
        shards = sorted(Path(path).glob("*.pkl"))
        for shard in shards:
            with open(shard, "rb") as f:
                header = pickle.load(f)
                if not (isinstance(header, dict) and header.get("version") == 2):
                    raise ValueError(f"{shard}: expected v2 compact shard")
                for _ in range(int(header["n_records"])):
                    self._tb_anchor_dicts.append(pickle.load(f))
        _random.Random(0).shuffle(self._tb_anchor_dicts)
        print(f"TB anchor archive: {len(self._tb_anchor_dicts)} games "
              f"from {len(shards)} shard(s) at {path}")

    def _inject_tb_anchor_games(self, n: int) -> None:
        """Inject ``n`` TB anchor games into the rolling buffer, cycling the
        archive (reshuffled each wrap). See _load_tb_anchor."""
        if not self._tb_anchor_dicts:
            return
        import random as _random
        for _ in range(n):
            if self._tb_anchor_cursor >= len(self._tb_anchor_dicts):
                _random.shuffle(self._tb_anchor_dicts)
                self._tb_anchor_cursor = 0
            d = self._tb_anchor_dicts[self._tb_anchor_cursor]
            self._tb_anchor_cursor += 1
            g = GameHistory.from_compact_dict(d, self.game)
            with self._buffer_lock:
                self.replay_buffer.save_game(g)
            self._tb_anchor_injected += 1
        self.writer.add_scalar(
            "injection/tb_anchor_total", self._tb_anchor_injected, self.global_step
        )
        tqdm.write(
            f"Step {self.global_step}: injected {n} TB anchor games "
            f"(total {self._tb_anchor_injected}, buffer {len(self.replay_buffer)})"
        )

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
            with self._buffer_lock:
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

    def _current_value_loss_weight(self) -> float:
        """Return the value-loss weight for this training step.

        When both ``value_loss_weight_warmstart`` and ``value_loss_weight_selfplay``
        are set, gate on ``self._injection_shards`` (the pool_alive signal): clean
        Stockfish targets get the warmstart weight, self-play-phase bootstrap
        targets get the selfplay weight. If either is ``None`` (older presets),
        fall back to the scalar ``value_loss_weight`` for back-compat.
        """
        w_warm = getattr(self.config, "value_loss_weight_warmstart", None)
        w_self = getattr(self.config, "value_loss_weight_selfplay", None)
        if w_warm is None or w_self is None:
            return self.config.value_loss_weight
        return w_warm if self._injection_shards else w_self

    # Parameter-name prefixes per sub-network, for splitting the total grad norm.
    _GRAD_GROUPS = {
        "repr": ("representation.",),
        "dyn_action_embed": ("dynamics.action_embedding.",),
        "dyn_body": ("dynamics.conv_in.", "dynamics.bn_in.", "dynamics.blocks."),
        "dyn_reward_head": ("dynamics.reward_head.",),
        "pred_policy_head": ("prediction.policy_head.",),
        "pred_value_head": ("prediction.value_head.",),
        "ssl_projection": ("projection.", "prediction_head."),
        "inverse_head": ("inverse_dynamics_head.",),
    }

    def _subnetwork_grad_norms(self) -> dict:
        """Per-sub-network pre-clip gradient L2 norms (call between unscale_ and clip)."""
        acc = {k: 0.0 for k in self._GRAD_GROUPS}
        for name, p in self.network.named_parameters():
            if p.grad is None:
                continue
            for key, prefixes in self._GRAD_GROUPS.items():
                if name.startswith(prefixes):
                    acc[key] += p.grad.detach().float().norm().item() ** 2
                    break
        return {k: v ** 0.5 for k, v in acc.items() if v > 0.0}

    @torch.no_grad()
    def _cross_action_cos(self, obs: torch.Tensor, actions: torch.Tensor) -> float:
        """Mean pairwise cosine of dynamics outputs for a fixed root across distinct
        actions. ~1.0 = action-blind dynamics; lower = action-aware. Diagnostic only."""
        probe_acts = torch.unique(actions[:, 0])[:12]
        n = int(probe_acts.numel())
        if n < 2:
            return float("nan")
        rh = self.network.representation(obs[:1])
        rh_n = rh.expand(n, *rh.shape[1:]).contiguous()
        dyn_out, _ = self.network.dynamics(rh_n, probe_acts)
        v = F.normalize(dyn_out.reshape(n, -1).float(), dim=-1)
        sim = v @ v.t()
        off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
        return off.mean().item()

    def _sample_batch_now(self):
        """Sample one training batch (CPU). Holds ``_buffer_lock`` so the read is
        consistent against concurrent ``save_game`` / ``update_priorities``.
        Called inline (prefetch off) or from the prefetch thread. Reads
        ``self.global_step`` for the step-dependent anneals (beta, warmstart/
        material schedules); when prefetched a step or two ahead that staleness is
        harmless. Returns ``(batch, game_indices, is_weights)``."""
        # Beta anneals from per_beta_init → 1.0 over training
        beta = self.config.per_beta_init + (1.0 - self.config.per_beta_init) * (
            self.global_step / max(1, self.config.training_steps)
        )
        with self._buffer_lock:
            return self.replay_buffer.sample_batch(
                self.config.batch_size,
                self.config.num_unroll_steps,
                self.config.td_steps,
                self.config.discount,
                self.game.action_space_size,
                alpha=self.config.per_alpha,
                beta=beta,
                value_head_type=getattr(self.config, "value_head_type", "support"),
                history_frames=getattr(self.config, "history_frames", 1),
                eval_to_wdl_alpha=getattr(self.config, "eval_to_wdl_alpha", 4.0),
                eval_to_wdl_beta=getattr(self.config, "eval_to_wdl_beta", 2.0),
                warmstart_sample_frac=get_warmstart_sample_frac(self.global_step, self.config),
                decisive_sample_frac=getattr(self.config, "decisive_sample_frac", 0.0),
                q_ratio=getattr(self.config, "q_ratio", 0.0),
                warmstart_q_ratio=getattr(self.config, "warmstart_q_ratio", None),
                selfplay_q_ratio=getattr(self.config, "selfplay_q_ratio", None),
                repetition_penalty=getattr(self.config, "repetition_penalty", 0.0),
                repetition_penalty_window=int(getattr(self.config, "repetition_penalty_window", 0)),
                repetition_penalty_decay=float(getattr(self.config, "repetition_penalty_decay", 0.0)),
                material_value_weight=get_material_value_weight(self.global_step, self.config),
                material_value_scale=float(getattr(self.config, "material_value_scale", 5.0)),
                tb_value_weight=float(getattr(self.config, "tb_value_weight", 0.0)),
                tb_moves_left_weight=float(getattr(self.config, "tb_moves_left_weight", 0.0)),
                tb_policy_weight=get_tb_policy_weight(self.global_step, self.config),
                tb_value_hard=bool(getattr(self.config, "tb_value_hard", False)),
                build_legal_masks=bool(getattr(self.config, "mask_illegal_policy", False)),
                build_material_target=bool(getattr(self.config, "use_material_head", False)),
            )

    def _next_batch(self):
        """Next ``(batch, game_indices, is_weights)`` — from the prefetch queue
        when the prefetch thread is running (re-raising any error it hit), else
        sampled inline. The timeout loop also detects a silently-dead thread."""
        if self.prefetch_enabled and self._prefetch_thread is not None:
            while True:
                try:
                    item = self._prefetch_queue.get(timeout=1.0)
                except queue.Empty:
                    if not self._prefetch_thread.is_alive():
                        if self._prefetch_error is not None:
                            raise self._prefetch_error
                        raise RuntimeError("batch-prefetch thread died")
                    continue
                if item is None:  # sentinel: thread hit an error
                    if self._prefetch_error is not None:
                        raise self._prefetch_error
                    raise RuntimeError("batch-prefetch thread stopped unexpectedly")
                return item
        return self._sample_batch_now()

    def _prefetch_loop(self):
        """Background producer: sample batches into the bounded queue until
        stopped. A full queue blocks the put (backpressure); the timeout lets the
        stop flag break the block at shutdown. Errors are surfaced to the consumer
        via a None sentinel + ``_prefetch_error``."""
        try:
            while not self._prefetch_stop.is_set():
                item = self._sample_batch_now()
                while not self._prefetch_stop.is_set():
                    try:
                        self._prefetch_queue.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:  # noqa: BLE001 — surface to the main thread
            self._prefetch_error = e
            traceback.print_exc()
            try:
                self._prefetch_queue.put_nowait(None)
            except Exception:
                pass

    def _start_prefetch(self):
        if not self.prefetch_enabled or self._prefetch_thread is not None:
            return
        self._prefetch_stop.clear()
        self._prefetch_error = None
        self._prefetch_queue = queue.Queue(maxsize=2)
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, name="batch-prefetch", daemon=True)
        self._prefetch_thread.start()

    def _stop_prefetch(self):
        if self._prefetch_thread is None:
            return
        self._prefetch_stop.set()
        # Unblock a put() that's waiting on a full queue so it can see the stop.
        try:
            while True:
                self._prefetch_queue.get_nowait()
        except queue.Empty:
            pass
        self._prefetch_thread.join(timeout=5.0)
        self._prefetch_thread = None
        self._prefetch_queue = None

    def _train_step(self) -> dict:
        """Single training step on a batch from the replay buffer."""
        self.network.train()

        batch, game_indices, is_weights = self._next_batch()

        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        target_values = batch["target_values"].to(self.device)
        target_rewards = batch["target_rewards"].to(self.device)
        target_policies = batch["target_policies"].to(self.device)
        is_weights_t = torch.tensor(is_weights, device=self.device)
        use_consistency = bool(getattr(self.config, "use_consistency_loss", False))
        single_frame_consistency = bool(getattr(self.config, "consistency_single_frame_target", False))
        use_inverse = bool(getattr(self.config, "use_inverse_dynamics_loss", False))
        inverse_weight = float(getattr(self.config, "inverse_dynamics_loss_weight", 1.0))
        # Newest-frame channel count, for slicing a single-frame consistency target
        # out of the T-frame stack (newest frame occupies the first num_planes channels).
        num_planes = getattr(self.game, "num_planes", None)
        target_obs_mask = batch["target_obs_mask"].to(self.device)             # (B, K+1)
        if use_consistency:
            target_observations = batch["target_observations"].to(self.device)  # (B, K+1, C, H, W)

        # Under Plain Gumbel MuZero the target is π' (softmax over π_net + σ(completedQ));
        # KL matches the paper's L_policy and reports 0 when π_net ≡ π'. Otherwise use
        # cross-entropy on raw visit counts (gradient is identical to KL up to entropy-of-target).
        policy_loss_fn = (
            self._policy_loss_kl if bool(getattr(self.config, "use_gumbel", False))
            else self._policy_loss
        )

        # Legal-move policy masking (chess: 99% of the action space is illegal).
        # When enabled, the standard full-softmax policy CE is kept and an
        # illegal-mass penalty is added on top. See config.mask_illegal_policy.
        mask_illegal = (bool(getattr(self.config, "mask_illegal_policy", False))
                        and "target_legal_masks" in batch)
        illegal_pen_w = float(getattr(self.config, "illegal_policy_penalty", 1.0))
        target_legal_masks = (batch["target_legal_masks"].to(self.device)  # (B, K+1, A)
                              if mask_illegal else None)

        is_warm = batch["is_warmstart"].to(self.device)

        # Per-sample value_loss_weight: warmstart games (clean Stockfish targets)
        # get stronger supervision than self-play games (noisy MCTS bootstraps).
        # Gated per-sample via is_warmstart so the weight is always correct
        # regardless of pool exhaustion state.
        w_warm = getattr(self.config, "value_loss_weight_warmstart", None)
        w_self = getattr(self.config, "value_loss_weight_selfplay", None)
        if w_warm is not None and w_self is not None:
            value_weight = torch.where(is_warm, w_warm, w_self)
        else:
            value_weight = self.config.value_loss_weight

        # Moves-left head (Lc0): aux CE on plies-to-end. Inert unless the head
        # exists AND the config enables it. Target is (B, K+1) plies-remaining.
        use_moves_left = (bool(getattr(self.config, "use_moves_left", False))
                          and getattr(self.network, "moves_left_head", None) is not None)
        moves_left_weight = float(getattr(self.config, "moves_left_loss_weight", 0.0))
        target_moves_left = (batch["target_moves_left"].to(self.device)
                             if use_moves_left and "target_moves_left" in batch else None)

        # Material-margin head (KataGo score-dist analogue): aux CE on STM material.
        # Inert unless the head exists AND the config enables it. Annealed weight.
        use_material_head = (bool(getattr(self.config, "use_material_head", False))
                             and getattr(self.network, "material_head", None) is not None)
        material_head_weight = (get_material_head_weight(self.global_step, self.config)
                                if use_material_head else 0.0)
        target_material = (batch["target_material"].to(self.device)
                           if use_material_head and "target_material" in batch else None)

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
            policy_logits_k0 = policy_logits  # save for entropy logging
            root_hidden_k0 = hidden  # save for in-loop cross-action-cos probe (detached later)

            # Per-sample losses at root (k=0) — always full weight
            lm0 = target_legal_masks[:, 0] if mask_illegal else None
            policy_loss, illegal_loss = self._policy_terms(
                policy_logits, target_policies[:, 0], lm0, policy_loss_fn)
            illegal_mass_root = illegal_loss.detach().mean().item()  # for logging
            value_loss = self._value_loss(value_logits, target_values[:, 0],
                                          self.network.value_support_size)
            reward_loss = torch.zeros(obs.shape[0], device=self.device)
            consistency_loss = torch.zeros(obs.shape[0], device=self.device)
            inverse_loss = torch.zeros(obs.shape[0], device=self.device)
            moves_left_loss = torch.zeros(obs.shape[0], device=self.device)
            if use_moves_left:
                moves_left_loss = self._moves_left_loss(
                    self.network.predict_moves_left(hidden), target_moves_left[:, 0])  # root, full weight
            material_loss = torch.zeros(obs.shape[0], device=self.device)
            if use_material_head:
                material_loss = self._material_loss(
                    self.network.predict_material(hidden), target_material[:, 0])  # root, full weight

            for k in range(self.config.num_unroll_steps):
                hidden_in = hidden  # h_k — the dynamics input at this unroll step
                hidden, reward_logits, policy_logits, value_logits = \
                    self.network.recurrent_inference_logits(hidden, actions[:, k])

                hidden.register_hook(lambda grad: grad * 0.5)

                mask_k = target_obs_mask[:, k + 1]
                lm_k = target_legal_masks[:, k + 1] if mask_illegal else None
                pl_k, im_k = self._policy_terms(
                    policy_logits, target_policies[:, k + 1], lm_k, policy_loss_fn)
                policy_loss = policy_loss + unroll_scale * pl_k * mask_k
                illegal_loss = illegal_loss + unroll_scale * im_k * mask_k
                value_loss = value_loss + unroll_scale * self._value_loss(value_logits, target_values[:, k + 1],
                                                                          self.network.value_support_size) * mask_k
                reward_loss = reward_loss + unroll_scale * self._reward_loss(reward_logits, target_rewards[:, k + 1],
                                                                             self.network.reward_support_size) * mask_k

                if use_consistency:
                    # SimSiam: online branch via dynamics (grad); target branch via representation
                    # of the actual future observation (stop-grad). Matches LightZero
                    # lzero/policy/efficientzero.py consistency loop.
                    target_obs_k = target_observations[:, k + 1]
                    if single_frame_consistency and num_planes is not None:
                        # Zero all but the newest frame so the target depends on the
                        # actual position at k+1, not the ~7/8 shared history frames.
                        sf = torch.zeros_like(target_obs_k)
                        sf[:, :num_planes] = target_obs_k[:, :num_planes]
                        target_obs_k = sf
                    dyn_proj = self.network.project(hidden, with_grad=True)
                    with torch.no_grad():
                        target_hidden = self.network.representation(target_obs_k)
                        tgt_proj = self.network.project(target_hidden, with_grad=False)
                    consistency_loss = consistency_loss + unroll_scale * (
                        _negative_cosine_similarity(dyn_proj, tgt_proj)
                        * target_obs_mask[:, k + 1]
                    )

                if use_inverse:
                    # ICM inverse model: recover a_k from (h_k, h_{k+1}). Gradient flows
                    # into both hidden states, forcing the dynamics output to encode the
                    # action. Masked past game end. CE over the full action space.
                    inv_logits = self.network.predict_inverse_action(hidden_in, hidden)
                    inverse_loss = inverse_loss + unroll_scale * (
                        F.cross_entropy(inv_logits, actions[:, k], reduction="none") * mask_k
                    )

                if use_moves_left:
                    moves_left_loss = moves_left_loss + unroll_scale * (
                        self._moves_left_loss(
                            self.network.predict_moves_left(hidden), target_moves_left[:, k + 1])
                        * mask_k
                    )

                if use_material_head:
                    material_loss = material_loss + unroll_scale * (
                        self._material_loss(
                            self.network.predict_material(hidden), target_material[:, k + 1])
                        * mask_k
                    )

            per_sample_loss = outer_scale * (
                policy_loss
                + value_weight * value_loss
                + reward_loss
                + self.config.consistency_loss_weight * consistency_loss
                + inverse_weight * inverse_loss
                + illegal_pen_w * illegal_loss
                + moves_left_weight * moves_left_loss
                + material_head_weight * material_loss
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
                "inverse_loss": float("nan"),
                "moves_left_loss": float("nan"),
                "material_loss": float("nan"),
                "grad_norm": float("nan"),
                "amp_scale": float(self.scaler.get_scale()),
                "value_mae": float("nan"),
                "value_target_std": float("nan"),
            }

        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        # Per-subnetwork grad norms (pre-clip), only on log steps. Splits the single
        # total grad_norm into where gradient actually lands — the diagnostic that
        # surfaces gradient starvation of the world-model body (zero-init heads, draw
        # basin) vs healthy flow into dynamics/representation/inverse head.
        log_now = (self.global_step % max(1, self.config.log_interval) == 0)
        subnet_grad = self._subnetwork_grad_norms() if log_now else {}
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
            # Histograms are expensive (one tensor copy per add_histogram per step
            # — many MB at batch=256). Throttle to once every N steps.
            log_hist = (self.global_step % max(1, getattr(self.config, "histogram_interval", 100)) == 0)
            # Logging-only diagnostics are computed only on log steps (each .item()/
            # .cpu() is a host sync that stalls the GPU). td_errors below is the one
            # exception — the PER priority update needs it every step.
            value_mae = value_target_std = float("nan")
            policy_entropy_pred = policy_entropy_target = float("nan")
            policy_loss_warm = policy_loss_self = float("nan")
            value_loss_warm = value_loss_self = float("nan")
            n_warm = 0
            if getattr(self.config, "value_head_type", "support") == "wdl":
                # WDL diagnostics: predicted scalar V = P(W) - P(L); target
                # scalar V = z[W] - z[L] (one-hot, so just +1/0/-1). MAE in
                # raw [-1, +1] space. target_std is the std of the target
                # scalar over the batch.
                from src.model.utils import wdl_to_scalar
                pred_v_scalar = wdl_to_scalar(
                    value_logits_k0.detach().float(),
                    draw_score=getattr(self.config, "draw_score", 0.0),
                )
                tgt_wdl = target_values[:, 0]  # (B, 3)
                true_v_scalar = (tgt_wdl[..., 0] - tgt_wdl[..., 2]).float()
                td_errors = (pred_v_scalar - true_v_scalar).abs().cpu().numpy()
                if log_now:
                    value_mae = (pred_v_scalar - true_v_scalar).abs().mean().item()
                    value_target_std = true_v_scalar.std(unbiased=False).item()
                # Histograms: predicted V scalar + each WDL component. A
                # collapsed value head shows pred_v_scalar histogram tightly
                # peaked at 0 with little spread; P_D >> P_W ≈ P_L is the
                # smoking-gun WDL distribution for draw bias.
                if log_hist:
                    pred_wdl = F.softmax(value_logits_k0.detach().float(), dim=-1)
                    self.writer.add_histogram("value/pred_v_scalar", pred_v_scalar, self.global_step)
                    self.writer.add_histogram("value/pred_p_w", pred_wdl[:, 0], self.global_step)
                    self.writer.add_histogram("value/pred_p_d", pred_wdl[:, 1], self.global_step)
                    self.writer.add_histogram("value/pred_p_l", pred_wdl[:, 2], self.global_step)
                    self.writer.add_histogram("value/target_v_scalar", true_v_scalar, self.global_step)
            else:
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
                if log_now:
                    value_mae = (pred_v_scalar - true_v_scalar).abs().mean().item()
                    value_target_std = true_v_scalar.float().std(unbiased=False).item()

            # Policy entropy at root (k=0). Tracks how sharp the model's prior
            # is (lower = sharper) and how sharp the training targets are. The
            # gap between them shows model fit; their absolute values show the
            # diffuse-vs-sharp regime. Both in nats over the full action space.
            if log_now:
                pred_probs = F.softmax(policy_logits_k0.detach().float(), dim=-1)
                policy_entropy_pred = -(
                    pred_probs * (pred_probs + 1e-12).log()
                ).sum(dim=-1).mean().item()
                tgt_probs = target_policies[:, 0].float()
                policy_entropy_target = -(
                    tgt_probs * (tgt_probs + 1e-12).log()
                ).sum(dim=-1).mean().item()

            # Per-stratum loss: warmstart vs self-play. Lever-2 health probes:
            #   - both staying high → anchor too small to absorb teaching
            #   - warmstart→0 while self-play diverges → overfitting on anchor
            #   - both falling together → working as intended
            # NaN when a stratum is empty in this batch (rare under stratified
            # sampling, common before pool exhaustion / after lever decays).
            if log_now:
                n_warm = int(is_warm.sum().item())
                n_sp = is_warm.numel() - n_warm
                policy_per_sample = (outer_scale * policy_loss).detach().float()
                value_per_sample = (outer_scale * value_loss).detach().float()
                policy_loss_warm = (policy_per_sample[is_warm].mean().item()
                                    if n_warm > 0 else float("nan"))
                policy_loss_self = (policy_per_sample[~is_warm].mean().item()
                                    if n_sp > 0 else float("nan"))
                value_loss_warm = (value_per_sample[is_warm].mean().item()
                                   if n_warm > 0 else float("nan"))
                value_loss_self = (value_per_sample[~is_warm].mean().item()
                                   if n_sp > 0 else float("nan"))
        with self._buffer_lock:
            self.replay_buffer.update_priorities(game_indices, td_errors, self.config.per_epsilon)

        # In-loop action-awareness probe (log steps only): cosine spread of the
        # dynamics output across distinct actions from a fixed root. ~1.0 = the
        # action-blind collapse; falling toward the ~0.6 inverse-trained regime is
        # the live signal that the dynamics is action-conditioned.
        cross_action_cos = self._cross_action_cos(obs, actions) if log_now else float("nan")

        result = {
            "total_loss": total_loss.item(),
            "policy_loss": (outer_scale * policy_loss.mean()).item(),
            # Drop the phase-dependent ``value_weight`` factor from the displayed
            # metric so the TB graph is comparable across the warmstart/self-play
            # phase boundary (the weight flips 1.0 → 0.25 there). The weighted
            # contribution to ``total_loss`` is preserved unchanged in total_loss.
            "value_loss": (outer_scale * value_loss.mean()).item(),
            "reward_loss": (outer_scale * reward_loss.mean()).item(),
            "consistency_loss": (outer_scale * self.config.consistency_loss_weight * consistency_loss.mean()).item(),
            "inverse_loss": (outer_scale * inverse_weight * inverse_loss.mean()).item(),
            "moves_left_loss": (outer_scale * moves_left_weight * moves_left_loss.mean()).item() if use_moves_left else float("nan"),
            "material_loss": (outer_scale * material_head_weight * material_loss.mean()).item() if use_material_head else float("nan"),
            "grad_norm": float(grad_norm),
            "amp_scale": float(self.scaler.get_scale()),
            "value_mae": value_mae,
            "value_target_std": value_target_std,
            "value_loss_weight": float(value_weight.mean()) if isinstance(value_weight, torch.Tensor) else float(value_weight),
            "policy_entropy_pred": policy_entropy_pred,
            "policy_entropy_target": policy_entropy_target,
            "policy_loss_warmstart": policy_loss_warm,
            "policy_loss_selfplay": policy_loss_self,
            "value_loss_warmstart": value_loss_warm,
            "value_loss_selfplay": value_loss_self,
            "batch_warmstart_frac": (n_warm / max(1, is_warm.numel())),
            "cross_action_cos": cross_action_cos,
            # Mean full-softmax probability mass on ILLEGAL moves at the root.
            # The thing legal-masking drives toward 0; ~0.17 unmasked at 22k.
            "policy_illegal_mass": illegal_mass_root,
        }
        # Per-sub-network grad norms (present only on log steps) → grad/<group>.
        for k, v in subnet_grad.items():
            result[f"gnorm_{k}"] = v
        return result

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
        # Reanalyze observations: rebuild the same T-frame stack the network
        # was trained on, so the model sees positions in the same encoding it
        # was trained on (otherwise input distribution shifts vs training).
        n_frames = getattr(self.config, "history_frames", 1)
        items = []
        for game in games:
            for pos in range(len(game.policies)):
                if pos < len(game.legal_actions_list):
                    obs_at_pos = game._stack_history(pos, n_frames)
                    items.append((obs_at_pos, game.legal_actions_list[pos], game, pos))

        if not items:
            return

        self.network.eval()
        # MCTS backend dispatch. Default: numpy BatchedMCTS (same path the
        # codebase shipped with). Opt-in: GPU TensorMCTS via the same factory
        # self-play uses — significantly faster per position. Gumbel root not
        # supported in TensorMCTS yet, so we hard-fail if the user combines flags.
        if getattr(self.config, "reanalyze_use_tensor_mcts", False):
            if use_gumbel:
                raise NotImplementedError(
                    "reanalyze_use_tensor_mcts=True is not compatible with "
                    "use_gumbel=True (TensorMCTS lacks Gumbel root)."
                )
            from ..mcts.tensor_mcts import TensorMCTS
            _dtype_map = {
                "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
            }
            hidden_dtype = _dtype_map[getattr(self.config, "tensor_mcts_hidden_dtype", "float32")]
            amp_str = getattr(self.config, "tensor_mcts_amp_dtype", None)
            amp_dtype = _dtype_map[amp_str] if amp_str else None
            batched_mcts = TensorMCTS(
                self.network, self.game, self.config,
                device=self.device,
                hidden_dtype=hidden_dtype,
                amp_dtype=amp_dtype,
                select_backend=getattr(self.config, "tensor_mcts_select_backend", "compile"),
                use_subtree_reuse=False,  # N/A: each reanalyze position is independent.
            )
        else:
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

            # Lock the in-place freshening: sample_batch (prefetch thread) reads
            # these same game.policies/root_values. The GPU run_batch above is
            # outside the lock; only the cheap write loop holds it.
            with self._buffer_lock:
                for (_, _, game, pos), root in zip(batch, roots):
                    if float(root.child_visits.sum()) > 0:
                        if use_gumbel:
                            # Plain Gumbel MuZero: π' = softmax(logits + σ(completedQ)) over legals.
                            _, action_probs = select_action_gumbel(
                                root, self.config, action_space_size
                            )
                            game.policies[pos] = _sparsify_policy(action_probs)
                        else:
                            # temperature=1.0 → raw visit-count normalization. Under
                            # Sampled MuZero the search prior was renormalized over σ,
                            # so raw visits already approximate Î_β π (Hubert 2021
                            # Proposed Modification).
                            _, action_probs = select_action(root, temperature=1.0)
                            # Pad to full action space size.
                            new_policy = np.zeros(action_space_size, dtype=np.float32)
                            new_policy[: len(action_probs)] = action_probs
                            game.policies[pos] = _sparsify_policy(new_policy)
                    game.root_values[pos] = root.value
                    positions_updated += 1

        # Mark each reanalyzed game deterministically (once per pass). Uses the
        # unique games that actually contributed positions to this batch, so a
        # game's reanalyze_count is the exact number of times its policy/value
        # targets were recomputed by a (newer) net — decoupling them from the
        # originally-played moves. Persisted via to_compact_dict.
        touched = {id(g): g for (_, _, g, _) in items}
        for g in touched.values():
            g.reanalyze_count += 1

        tqdm.write(f"Step {self.global_step}: reanalyze done ({positions_updated} positions updated, "
                   f"{len(touched)} games marked)")
        self.writer.add_scalar("reanalyze/positions_updated", positions_updated, self.global_step)

    def _policy_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-sample cross-entropy between predicted policy and MCTS visit distribution."""
        return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)

    def _policy_terms(self, logits, targets, legal_mask, base_loss_fn):
        """Per-sample (policy_ce, illegal_mass) for one unroll step.

        legal_mask is None  -> standard full-softmax CE, illegal_mass = 0.
        legal_mask (B, A)   -> the SAME standard full-softmax CE (which learns
                                legality for free via the shared normalizer:
                                raising legal logits drains probability off illegal
                                ones), PLUS the full-softmax probability mass on
                                illegal moves returned separately as a penalty
                                signal that drives illegal_mass below the CE's
                                natural floor — incl. at latent leaves we can't mask.

        We deliberately do NOT renormalize the softmax over legal moves: that is
        shift-invariant over the legal logits (adding a constant to all of them
        leaves the loss unchanged) and gives illegal logits zero gradient, so it
        cannot teach legality. Formally full_CE = masked_CE - log P(legal), i.e.
        renormalizing just drops the -log P(legal) legality term; the penalty here
        is extra suppression on top of the full CE's own legality learning.
        """
        ce = base_loss_fn(logits, targets)
        if legal_mask is None:
            return ce, logits.new_zeros(logits.shape[0])
        illegal_mass = (F.softmax(logits, dim=1) * (1.0 - legal_mask)).sum(dim=1)
        return ce, illegal_mass

    def _policy_loss_kl(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-sample KL(π'_target || π_net) — paper Eq. 12.

        Gradient matches cross-entropy up to H(target) (constant in network params),
        but the reported loss value is more interpretable (→ 0 as π_net → π').
        F.kl_div handles ``target == 0`` safely via the 0·log(0) = 0 convention.
        """
        log_p = F.log_softmax(logits, dim=1)
        return F.kl_div(log_p, targets, reduction="none").sum(dim=1)

    def _value_loss(self, logits: torch.Tensor, target: torch.Tensor,
                    support_size: int) -> torch.Tensor:
        """Per-sample cross-entropy for value prediction.

        Two paths depending on ``self.config.value_head_type``:
        - 'support' (default): scalar target → categorical via scalar_to_support → CE
          against logits of shape (B, 2*support_size+1).
        - 'wdl': target is already a (B, 3) probability distribution; CE directly
          against logits of shape (B, 3). support_size is unused.
        """
        if getattr(self.config, "value_head_type", "support") == "wdl":
            # target shape: (B, 3) — already a probability distribution.
            return -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
        # Support-head path (original behaviour).
        target_scalar = target
        if self.config.use_scalar_transform:
            transformed = scalar_transform(target_scalar)
        else:
            transformed = target_scalar * self.config.value_target_scale
        target_dist = scalar_to_support(transformed, support_size).to(logits.device)
        return -(target_dist * F.log_softmax(logits, dim=1)).sum(dim=1)

    def _moves_left_loss(self, logits: torch.Tensor, target_scalar: torch.Tensor) -> torch.Tensor:
        """Per-sample CE for the moves-left head. ``target_scalar``: (B,) plies-
        remaining (>=0). Always applies the scalar_transform (compressed support →
        fine resolution near 0, where 'win faster' actually matters). POV-independent,
        so no negamax flip. Mirrors _value_loss's support path."""
        transformed = scalar_transform(target_scalar)
        target_dist = scalar_to_support(transformed, self.network.moves_left_support_size).to(logits.device)
        return -(target_dist * F.log_softmax(logits, dim=1)).sum(dim=1)

    def _material_loss(self, logits: torch.Tensor, target_scalar: torch.Tensor) -> torch.Tensor:
        """Per-sample CE for the material-margin head. ``target_scalar``: (B,)
        STM-relative material margin in pawns (signed). Always applies the
        scalar_transform (compressed support → fine resolution near 0). STM-
        relative target, so no negamax flip. Mirrors _moves_left_loss."""
        transformed = scalar_transform(target_scalar)
        target_dist = scalar_to_support(transformed, self.network.material_head_support_size).to(logits.device)
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
        value_keys = {"value_mae", "value_target_std", "value_loss_weight"}
        policy_keys = {"policy_entropy_pred", "policy_entropy_target", "policy_illegal_mass"}
        train_keys = {"batch_warmstart_frac"}
        for key, value in loss_info.items():
            # Skip NaN universally (TB grays curves on NaN; e.g. cross_action_cos /
            # gnorm_* / strata are only populated on log steps or non-empty strata).
            if isinstance(value, float) and value != value:  # NaN
                continue
            if key.startswith("gnorm_"):
                # Per-sub-network pre-clip grad norm → grad/<group>.
                self.writer.add_scalar(f"grad/{key[len('gnorm_'):]}", value, step)
            elif key == "cross_action_cos":
                self.writer.add_scalar("dynamics/cross_action_cos", value, step)
            elif key in ("grad_norm", "amp_scale") or key in train_keys:
                self.writer.add_scalar(f"train/{key}", value, step)
            elif key in value_keys:
                self.writer.add_scalar(f"value/{key[len('value_'):]}", value, step)
            elif key in policy_keys:
                self.writer.add_scalar(f"policy/{key[len('policy_'):]}", value, step)
            else:
                # Per-stratum keys (policy_loss_warmstart etc.) and the rest
                # land in loss/ alongside the aggregate counterparts so they
                # graph next to each other in TB.
                self.writer.add_scalar(f"loss/{key}", value, step)
        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], step)
        self.writer.add_scalar("train/per_beta", self.config.per_beta_init + (1.0 - self.config.per_beta_init) * (
            step / max(1, self.config.training_steps)
        ), step)
        # Annealed material weights (the curriculum schedule) — so the decay is
        # visible alongside loss/material_loss. Only when material is active.
        self.writer.add_scalar("train/warmstart_sample_frac",
                               get_warmstart_sample_frac(step, self.config), step)
        if getattr(self.config, "material_value_weight", 0.0) > 0 or getattr(self.config, "use_material_head", False):
            self.writer.add_scalar("material/value_blend_weight",
                                   get_material_value_weight(step, self.config), step)
            if getattr(self.config, "use_material_head", False):
                self.writer.add_scalar("material/head_loss_weight",
                                       get_material_head_weight(step, self.config), step)
        # Training throughput across this log window (steps/sec).
        now = time.monotonic()
        if self._last_log_t is not None and self._last_log_step is not None and now > self._last_log_t:
            sps = (step - self._last_log_step) / (now - self._last_log_t)
            if sps > 0:
                self.writer.add_scalar("train/steps_per_sec", sps, step)
        self._last_log_t = now
        self._last_log_step = step

    def _save_checkpoint(self, step: int):
        torch.save({
            "step": step,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "injection_loaded": self._injection_loaded,
        }, self.checkpoint_dir / f"checkpoint_{step}.pt")

        # save_buffer=False short-circuits the .buf write entirely. Used when
        # the compact format is unstable (chess preset 2026-05-07 onward).
        # Resume falls through to empty-buffer cold-start — see
        # load_checkpoint's "No saved buffer found" path.
        if not getattr(self.config, "save_buffer", True):
            tqdm.write(
                f"Step {step}: skipped buffer pickle "
                f"(save_buffer=False; resume will cold-start buffer)"
            )
            return

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

    def load_body_warmstart(self, checkpoint_path: str):
        """Warm-start the BODY (+ any shape-matching heads) from a checkpoint whose
        architecture differs from the current model — e.g. after widening the
        moves-left head's input. Loads model weights with strict=False (matching
        keys transfer: representation/dynamics/policy/value; changed heads keep their
        fresh init), KEEPS the replay buffer (data is model-independent) and the step
        counter (so anneals/schedules continue), but uses a FRESH optimizer/scheduler
        (stale moments for reinitialized heads would corrupt their early training)."""
        from src.config import MuZeroConfig
        torch.serialization.add_safe_globals([MuZeroConfig])
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        ckpt_sd = ckpt["model_state_dict"]
        model_sd = self.network.state_dict()
        # Transfer only keys present in BOTH with MATCHING shape. Shape-changed keys
        # (e.g. a widened head's convs) error even under strict=False, so filter them
        # out explicitly; they keep their fresh init.
        transfer = {k: v for k, v in ckpt_sd.items()
                    if k in model_sd and model_sd[k].shape == v.shape}
        fresh = sorted({k.split(".")[0] for k in model_sd if k not in transfer})
        self.network.load_state_dict(transfer, strict=False)
        print(f"Body warm-start from {checkpoint_path}:")
        print(f"  transferred {len(transfer)}/{len(model_sd)} tensors; "
              f"FRESH (absent or shape-changed): {fresh or 'none'}")
        self.global_step = int(ckpt.get("step", 0))
        self._injection_loaded = int(ckpt.get("injection_loaded", 0))
        buf_path = Path(checkpoint_path).with_suffix(".buf")
        if buf_path.exists():
            try:
                self.replay_buffer.load(buf_path, game=self.game)
                print(f"  loaded replay buffer ({len(self.replay_buffer)} games)")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"  WARNING: {buf_path.name} corrupt ({type(e).__name__}); empty buffer")
                self.replay_buffer.buffer = []
                self.replay_buffer._priorities = []
        else:
            print("  no saved buffer found — starting with empty buffer")
        print(f"  resuming at step {self.global_step} with a FRESH optimizer/scheduler")

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
        game, per move. Per-game ply observation history is tracked so the
        T-frame history-encoded input passed to MCTS matches what the network
        sees during training (sample-time stacking of the same per-ply obs).
        """
        import random
        from ..mcts.mcts import BatchedMCTS, select_action, select_action_gumbel
        from .replay_buffer import stack_with_history

        use_gumbel = bool(getattr(self.config, "use_gumbel", False))
        n_frames = getattr(self.config, "history_frames", 1)
        batched = BatchedMCTS(self.network, self.game, self.config, self.device)
        states = [self.game.reset() for _ in range(n_games)]
        ply_obs_history: list[list] = [[] for _ in range(n_games)]

        while any(not s.done for s in states):
            for i, s in enumerate(states):
                if not s.done and s.current_player == -1:
                    # Capture pre-random-move obs into history so the next agent
                    # turn sees the right T-frame window.
                    ply_obs_history[i].append(self.game.to_tensor(s))
                    a = random.choice(self.game.legal_actions(s))
                    states[i], _, _ = self.game.step(s, a)

            agent_idx = [i for i, s in enumerate(states) if not s.done and s.current_player == 1]
            if not agent_idx:
                continue
            single_frames = [self.game.to_tensor(states[i]) for i in agent_idx]
            obs_list = [
                stack_with_history(single_frames[k], ply_obs_history[i], n_frames)
                for k, i in enumerate(agent_idx)
            ]
            # Append current obs to history AFTER stacking so the next iteration
            # sees this ply as "prior".
            for k, i in enumerate(agent_idx):
                ply_obs_history[i].append(single_frames[k])
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
