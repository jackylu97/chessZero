"""Benchmark: self-play wall with the deferred relabel at workers=1 (serial — the
same total probe work the OLD inline path did) vs workers=8 (pooled). Same seeds +
torch seed → identical games, so the only difference is relabel parallelism.

Run: PYTHONPATH=. .venv/bin/python scripts/_bench_deferred_relabel.py
"""
import time, random
import torch

from src.config import get_config
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.training.self_play import play_games_parallel_gpu_resident

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NG = 64


def build():
    cfg = get_config("chess_small")
    cfg.tb_root_probe = True
    cfg.tb_path = "data/syzygy"; cfg.tb_gaviota_path = "data/gaviota"
    cfg.tb_max_pieces = 5
    cfg.tb_policy_weight = 1.0; cfg.tb_policy_temp = 0.15; cfg.tb_value_dtz_shape = 0.5
    cfg.tb_steer_policy = False
    cfg.max_plies = 80
    cfg.num_simulations = 200
    game = ChessGame()
    net = MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size, hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=game.board_size[0], input_w=game.board_size[1], fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size, reward_support_size=cfg.reward_support_size,
        action_embed_dim=cfg.action_embed_dim, use_consistency_loss=cfg.use_consistency_loss,
        proj_hid=cfg.proj_hid, proj_out=cfg.proj_out, pred_hid=cfg.pred_hid, pred_out=cfg.pred_out,
        use_scalar_transform=cfg.use_scalar_transform, value_target_scale=cfg.value_target_scale,
        value_head_type=cfg.value_head_type, draw_score=getattr(cfg, "draw_score", 0.0),
        policy_head_type=cfg.policy_head_type, use_moves_left=cfg.use_moves_left,
        moves_left_support_size=cfg.moves_left_support_size,
        use_material_head=cfg.use_material_head,
        material_head_support_size=cfg.material_head_support_size).to(DEV).eval()
    return cfg, net


def run(cfg, net, workers, fens, warmup=False):
    cfg.tb_relabel_workers = workers
    random.seed(7); torch.manual_seed(7)
    n = 8 if warmup else NG
    t0 = time.perf_counter()
    play_games_parallel_gpu_resident(net, cfg, num_games=n, device=DEV,
                                     training_step=0, start_fens=fens[:n])
    return time.perf_counter() - t0


def main():
    cfg, net = build()
    seeds = [ln.strip() for ln in open("data/endgame_seeds.txt") if ln.strip()]
    random.seed(3); random.shuffle(seeds)
    fens = seeds[:NG]

    # prime the spawn pool (its ~1-2s startup is one-time across a whole run)
    run(cfg, net, 8, fens, warmup=True)

    t1 = run(cfg, net, 1, fens)
    t8 = run(cfg, net, 8, fens)
    print(f"\nself-play batch ({NG} seeded games, 200 sims, <=80 plies):")
    print(f"  workers=1 (serial deferred): {t1:6.2f}s")
    print(f"  workers=8 (pooled deferred): {t8:6.2f}s")
    print(f"  speedup: {t1 / max(t8, 1e-9):.2f}x")


if __name__ == "__main__":
    main()
