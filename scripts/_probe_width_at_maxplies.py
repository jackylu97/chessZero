"""Realistic self-play memory + game-length probe at the PRODUCTION max_plies.

The earlier ceiling probe used max_plies=32 and badly under-counted: GPU memory
in the resident sweep GROWS with plies (per-ply trajectory storage), so peak is
~ width * plies. This runs FULL seed-like rounds (cold random net -> long games)
at the real max_plies with compile ON + expandable_segments, reporting true peak
and mean game length per width. Pick the largest width whose peak leaves margin.
"""
import sys, time, torch
sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.training.self_play import run_self_play

MAX_PLIES = 750
WIDTHS = [512, 768]

config = get_config("chess_small")
config.device = "cuda"
config.use_gpu_chess = True
config.use_tensor_mcts = True
config.use_gpu_resident_self_play = True
config.tensor_mcts_compile_net = True       # match prod (--tensor-mcts-compile-net)
config.mask_illegal_policy = True
config.max_plies = MAX_PLIES
config.num_simulations = 200
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
print(f"GPU total: {total_gb:.1f} GB | max_plies={MAX_PLIES} | compile=ON\n", flush=True)

for w in WIDTHS:
    config.num_parallel_games = w
    config.num_self_play_games = w
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        games = run_self_play(network, game, config, w, "cuda",
                              show_progress=False, training_step=0)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        resv = torch.cuda.max_memory_reserved() / 1e9
        L = sum(len(g) for g in games) / len(games)
        print(f"  W={w:5d}  OK   peak={peak:6.2f} GB  reserved={resv:6.2f} GB  "
              f"meanlen={L:6.1f}  round={dt:5.0f}s  margin={total_gb-resv:5.2f}GB", flush=True)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print(f"  W={w:5d}  OOM", flush=True)
        else:
            raise
    torch.cuda.empty_cache()
