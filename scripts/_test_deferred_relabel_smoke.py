"""Integration smoke test: a real deferred (pooled) self-play batch on seeded
endgames must run and populate the GameHistory TB target fields.

Run: PYTHONPATH=. .venv/bin/python scripts/_test_deferred_relabel_smoke.py
"""
import sys, random
import torch, numpy as np

from src.config import get_config
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.training.self_play import play_games_parallel_gpu_resident

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NG = 16
random.seed(1); torch.manual_seed(1)


def main():
    cfg = get_config("chess_small")
    cfg.tb_root_probe = True
    cfg.tb_path = "data/syzygy"
    cfg.tb_gaviota_path = "data/gaviota"
    cfg.tb_max_pieces = 5
    cfg.tb_policy_weight = 1.0          # → want_policy=True (exercise the per-move classify)
    cfg.tb_policy_temp = 0.15
    cfg.tb_value_dtz_shape = 0.5
    cfg.tb_steer_policy = False         # → defer_relabel=True
    cfg.tb_relabel_workers = 4          # pooled
    cfg.max_plies = 40
    cfg.num_simulations = 32            # fast

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

    seeds = [ln.strip() for ln in open("data/endgame_seeds.txt") if ln.strip()]
    random.shuffle(seeds)
    start_fens = seeds[:NG]

    hist = play_games_parallel_gpu_resident(
        net, cfg, num_games=NG, device=DEV, training_step=0, start_fens=start_fens)

    assert len(hist) == NG, f"expected {NG} histories, got {len(hist)}"
    n_val = sum(1 for h in hist if getattr(h, "tablebase_values", None))
    n_ml = sum(1 for h in hist if getattr(h, "tablebase_moves_left", None))
    n_pol = sum(1 for h in hist if getattr(h, "tablebase_policy", None))
    # sanity: among games with a value-target list, values are finite somewhere
    finite_val = 0
    for h in hist:
        tv = getattr(h, "tablebase_values", None)
        if tv and any(np.isfinite(v) for v in tv):
            finite_val += 1
    # policy entries are (idx[], w[]) summing ~1
    pol_ok = True
    for h in hist:
        tp = getattr(h, "tablebase_policy", None)
        if not tp:
            continue
        for p in tp:
            if p is None:
                continue
            idx, w = p
            if abs(float(np.sum(w)) - 1.0) > 1e-4:
                pol_ok = False

    print(f"histories: {len(hist)}")
    print(f"with tablebase_values:     {n_val}/{NG}  (finite: {finite_val})")
    print(f"with tablebase_moves_left: {n_ml}/{NG}")
    print(f"with tablebase_policy:     {n_pol}/{NG}")
    print(f"policy rows sum to 1: {pol_ok}")
    # seeds are in-TB from ply 0, so essentially all seeded games carry value targets.
    ok = (n_val >= NG - 1 and finite_val >= NG - 1 and n_pol >= 1 and pol_ok)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
