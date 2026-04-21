"""Main training entrypoint for MuZero."""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import get_config
from src.games.tictactoe import TicTacToe
from src.model.muzero_net import MuZeroNetwork
from src.training.run_id import generate_run_id
from src.training.trainer import MuZeroTrainer


GAMES = {
    "tictactoe": TicTacToe,
}

# Lazy imports for games that may not be needed
def get_game(name: str):
    if name in GAMES:
        return GAMES[name]()
    if name == "connect4":
        from src.games.connect4 import Connect4
        return Connect4()
    if name == "chess":
        from src.games.chess import ChessGame
        return ChessGame()
    if name == "checkers":
        from src.games.checkers import Checkers
        return Checkers()
    raise ValueError(f"Unknown game: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train MuZero")
    parser.add_argument("--game", type=str, default="tictactoe",
                        choices=["tictactoe", "connect4", "chess", "checkers"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID (default: auto-generate YYYY_MM_DD_NNNN). "
                             "Pass an existing ID to continue writing into that run's dirs.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--sample-k", type=int, default=None,
                        help="Sampled MuZero K. None = deterministic top-K (legacy).")
    parser.add_argument("--use-gumbel", action="store_true",
                        help="Enable Plain Gumbel MuZero (Danihelka 2022) at root.")
    parser.add_argument("--gumbel-m", type=int, default=None,
                        help="gumbel_num_considered (m for Sequential Halving).")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="Override config.eval_interval (steps between evals).")
    parser.add_argument("--warmstart-path", type=str, default=None,
                        help="Directory or glob of .pkl shards (list[GameHistory]) "
                             "to pre-load into the replay buffer before training. "
                             "Games with populated external_values use those as value "
                             "targets (bypassing td_steps/game_outcome). Typically "
                             "produced by scripts/generate_stockfish_games.py.")
    parser.add_argument("--root-heavy-loss", action="store_true",
                        help="Weight the root prediction at 1.0 and each unroll step "
                             "at 1/K (MuZero paper / muzero-general convention). "
                             "Default is uniform 1/(K+1) per step (LightZero-like).")
    parser.add_argument("--max-buf-save-games", type=int, default=None,
                        help="Cap the number of most-recent self-play games persisted "
                             "to .buf per checkpoint. In-memory buffer is unaffected. "
                             "Default: no cap.")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    config = get_config(args.game)
    config.device = device
    if args.steps is not None:
        config.training_steps = args.steps
    if args.sample_k is not None:
        config.sample_k = args.sample_k
    if args.use_gumbel:
        config.use_gumbel = True
    if args.gumbel_m is not None:
        config.gumbel_num_considered = args.gumbel_m
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.root_heavy_loss:
        config.use_root_heavy_loss = True
    if args.max_buf_save_games is not None:
        config.max_buf_save_games = args.max_buf_save_games

    # Use CPU AMP settings appropriately
    if device == "cpu":
        config.use_amp = False

    game = get_game(args.game)

    run_id = args.run_id or generate_run_id(
        Path(args.checkpoints_dir) / args.game,
        Path(args.log_dir) / args.game,
    )
    print(f"Run ID: {run_id}")
    print(f"  Checkpoints: {Path(args.checkpoints_dir) / args.game / run_id}")
    print(f"  TensorBoard: {Path(args.log_dir) / args.game / run_id}")

    network = MuZeroNetwork(
        observation_channels=game.num_planes,
        action_space_size=game.action_space_size,
        hidden_planes=config.hidden_planes,
        num_blocks=config.num_residual_blocks,
        latent_h=config.latent_h,
        latent_w=config.latent_w,
        input_h=game.board_size[0],
        input_w=game.board_size[1],
        fc_hidden=config.fc_hidden,
        value_support_size=config.value_support_size,
        reward_support_size=config.reward_support_size,
        use_consistency_loss=config.use_consistency_loss,
        proj_hid=config.proj_hid,
        proj_out=config.proj_out,
        pred_hid=config.pred_hid,
        pred_out=config.pred_out,
        use_scalar_transform=config.use_scalar_transform,
        value_target_scale=config.value_target_scale,
    )

    trainer = MuZeroTrainer(
        config, game, network, run_id,
        device=device,
        log_dir=args.log_dir,
        checkpoints_dir=args.checkpoints_dir,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if args.warmstart_path:
        import glob
        p = args.warmstart_path
        if os.path.isdir(p):
            shard_paths = sorted(glob.glob(os.path.join(p, "*.pkl")))
        else:
            shard_paths = sorted(glob.glob(p))
        if not shard_paths:
            raise FileNotFoundError(f"No .pkl shards found under --warmstart-path: {p}")
        print(f"Warmstart: loading {len(shard_paths)} shard(s) from {p}")
        n_loaded = trainer.replay_buffer.load_warmstart_games(shard_paths)
        print(f"Warmstart: pre-seeded {n_loaded} games "
              f"(buffer now {len(trainer.replay_buffer)}/{config.replay_buffer_size})")

    trainer.train()


if __name__ == "__main__":
    main()
