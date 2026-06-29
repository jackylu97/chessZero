"""Self-play THROUGHPUT vs parallel width — find the GPU saturation point.

Times a real GPU-resident self-play sweep (real sims=200, capped max_plies for
timing) at increasing widths. Throughput = games*plies/sec. It rises with width
until the GPU compute-saturates, then plateaus — past that, wider only inflates
per-round latency (and, at fixed 1:1, cuts policy-iteration cycles).
"""
import sys, time, torch
sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.training.self_play import run_self_play

WIDTHS = [256, 512, 1024, 2048, 3072]
PLIES = 24                                  # short, fixed loop for timing

config = get_config("chess_small")
config.device = "cuda"
config.use_gpu_chess = True
config.use_tensor_mcts = True
config.use_gpu_resident_self_play = True
config.tensor_mcts_compile_net = False      # note: prod compile is ~1.4x faster
config.mask_illegal_policy = True
config.max_plies = PLIES
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

# warmup (cudnn autotune) at smallest width
config.num_parallel_games = config.num_self_play_games = 256
run_self_play(network, game, config, 256, "cuda", show_progress=False)
torch.cuda.synchronize()

print(f"{'width':>6} {'sec':>7} {'ply-sweeps/s':>13} {'rel-thru':>9}", flush=True)
base = None
for w in WIDTHS:
    config.num_parallel_games = config.num_self_play_games = w
    torch.cuda.synchronize(); t0 = time.time()
    run_self_play(network, game, config, w, "cuda", show_progress=False)
    torch.cuda.synchronize(); dt = time.time() - t0
    thru = w * PLIES / dt                    # (games in flight * plies) / sec
    if base is None: base = thru
    print(f"{w:>6} {dt:>7.2f} {thru:>13.0f} {thru/base:>8.2f}x", flush=True)
