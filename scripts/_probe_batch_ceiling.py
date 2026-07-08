"""Find the max training batch_size that fits on this GPU for chess_small.

Drives the REAL Trainer._train_step (full K-unroll loss, AMP, legal masks) against
an existing buffer at increasing batch sizes, reporting peak CUDA memory / OOM.
"""
import sys, gc
import torch
sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.training.trainer import MuZeroTrainer

BUF = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_25000.buf"
CANDIDATES = [256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192]

config = get_config("chess_small")
config.device = "cuda"
config.mask_illegal_policy = True
config.value_head_init_std = 0.01
config.replay_buffer_size = 10000          # so the load isn't truncated
config.compile_network = False
game = ChessGame()

network = MuZeroNetwork(
    observation_channels=game.num_planes * getattr(config, "history_frames", 1),
    action_space_size=game.action_space_size,
    hidden_planes=config.hidden_planes, num_blocks=config.num_residual_blocks,
    latent_h=config.latent_h, latent_w=config.latent_w,
    input_h=game.board_size[0], input_w=game.board_size[1],
    fc_hidden=config.fc_hidden, value_support_size=config.value_support_size,
    reward_support_size=config.reward_support_size,
    action_embed_dim=getattr(config, "action_embed_dim", 16),
    use_consistency_loss=config.use_consistency_loss,
    proj_hid=config.proj_hid, proj_out=config.proj_out,
    pred_hid=config.pred_hid, pred_out=config.pred_out,
    use_scalar_transform=config.use_scalar_transform,
    value_target_scale=config.value_target_scale,
    value_head_type=getattr(config, "value_head_type", "support"),
    draw_score=getattr(config, "draw_score", 0.0),
    value_head_init_std=getattr(config, "value_head_init_std", 0.0),
    use_inverse_dynamics_loss=getattr(config, "use_inverse_dynamics_loss", False),
    inverse_dynamics_hidden=getattr(config, "inverse_dynamics_hidden", 256),
    policy_head_type=getattr(config, "policy_head_type", "flat"),
    use_moves_left=getattr(config, "use_moves_left", False),
    moves_left_support_size=getattr(config, "moves_left_support_size", 10),
)

trainer = MuZeroTrainer(config, game, network, "probe_batch_ceiling", device="cuda",
                        log_dir="/tmp/probe_runs", checkpoints_dir="/tmp/probe_ckpts")
n = trainer.replay_buffer.load(BUF, game=game)
print(f"loaded buffer: {len(trainer.replay_buffer)} games", flush=True)

total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU total: {total_gb:.1f} GB\n", flush=True)

last_ok = None
for bs in CANDIDATES:
    config.batch_size = bs
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(3):                 # warm + steady; cudnn.benchmark picks algos
            trainer._train_step()
        peak = torch.cuda.max_memory_allocated() / 1e9
        resv = torch.cuda.max_memory_reserved() / 1e9
        print(f"  bs={bs:5d}  OK   peak_alloc={peak:6.2f} GB  reserved={resv:6.2f} GB", flush=True)
        last_ok = bs
    except torch.cuda.OutOfMemoryError as e:
        print(f"  bs={bs:5d}  OOM  ({str(e)[:60]})", flush=True)
        break
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  bs={bs:5d}  OOM  (RuntimeError)", flush=True)
            break
        raise
    finally:
        gc.collect(); torch.cuda.empty_cache()

print(f"\nLargest batch that fit: {last_ok}", flush=True)
