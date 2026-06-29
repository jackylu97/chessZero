"""Find the max num_parallel_games for GPU-resident self-play on this GPU.

Runs the REAL run_self_play GPU-resident path at increasing parallel widths
(real num_simulations=200 for true MCTS-tree size; short max_plies for speed),
reporting peak CUDA memory / OOM. Self-play has the card to itself at run time
(trainer empty_cache()s before each round), so the ceiling is self-play's own
footprint against ~32 GB.
"""
import sys, gc
import torch
sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.training.self_play import run_self_play

WIDTHS = [256, 512, 1024, 1536, 2048, 2560, 3072]

config = get_config("chess_small")
config.device = "cuda"
config.use_gpu_chess = True
config.use_tensor_mcts = True
config.use_gpu_resident_self_play = True
config.tensor_mcts_compile_net = False   # memory ~ tree size, compile-independent
config.mask_illegal_policy = True
config.max_plies = 32                     # short loop; peak mem plateaus after a few plies
config.num_simulations = 200              # real sim count -> real tree footprint
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
).to("cuda").eval()

total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU total: {total_gb:.1f} GB\n", flush=True)

last_ok = None
for w in WIDTHS:
    config.num_parallel_games = w
    config.num_self_play_games = w          # one sweep of w games
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        run_self_play(network, game, config, w, "cuda",
                      show_progress=False, training_step=0)
        peak = torch.cuda.max_memory_allocated() / 1e9
        resv = torch.cuda.max_memory_reserved() / 1e9
        print(f"  num_parallel={w:5d}  OK   peak_alloc={peak:6.2f} GB  reserved={resv:6.2f} GB", flush=True)
        last_ok = w
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print(f"  num_parallel={w:5d}  OOM", flush=True)
            break
        raise
    finally:
        gc.collect(); torch.cuda.empty_cache()

print(f"\nLargest parallel width that fit: {last_ok}", flush=True)
