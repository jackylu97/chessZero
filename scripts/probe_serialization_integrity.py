"""Data-integrity probe for GameHistory serialization.

Three checks:
  A. Round-trip to_compact_dict / from_compact_dict: do all scalar/array fields
     survive bit-exact?
  B. GPU-generated obs vs CPU-reconstructed obs: the .buf stores ACTIONS only;
     on load, observations are rebuilt by replaying actions through the CPU
     ChessGame.to_tensor(). If that disagrees with the GPU to_tensor_batch()
     that ACTUALLY generated the game, every loaded obs is silently corrupt.
  C. make_target stability: does a reloaded .buf game produce the same
     make_target output as the game in memory?
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config
from src.games.chess import ChessGame
from src.games.chess_gpu import GpuChessGame
from src.training.replay_buffer import GameHistory, ReplayBuffer, _iter_shard_games


def load_one_buf_game(path, game, n=3):
    """Load up to n games from a v3 .buf shard."""
    out = []
    with open(path, "rb") as f:
        header = pickle.load(f)
        assert isinstance(header, dict) and header.get("version") in (2, 3), header
        ver = header["version"]
        for _ in range(min(n, header["n_records"])):
            record, prio = pickle.load(f)
            if ver == 3:
                record = GameHistory.from_compact_dict(record, game)
            out.append((record, prio))
    return out, header


def check_A_roundtrip(gh, game):
    """to_compact_dict -> from_compact_dict: assert fields survive."""
    d = gh.to_compact_dict()
    gh2 = GameHistory.from_compact_dict(d, game)
    fails = []
    if gh.actions != gh2.actions:
        fails.append("actions mismatch")
    if not np.allclose(gh.rewards, gh2.rewards, atol=0, rtol=0):
        fails.append(f"rewards mismatch max|d|={np.max(np.abs(np.array(gh.rewards)-np.array(gh2.rewards)))}")
    if not np.allclose(gh.root_values, gh2.root_values, atol=0, rtol=0):
        fails.append(f"root_values mismatch max|d|={np.max(np.abs(np.array(gh.root_values)-np.array(gh2.root_values)))}")
    if gh.game_outcome != gh2.game_outcome:
        fails.append(f"game_outcome {gh.game_outcome} != {gh2.game_outcome}")
    if gh.draw_by_repetition != gh2.draw_by_repetition:
        fails.append("draw_by_repetition mismatch")
    if gh.draw_by_no_progress != gh2.draw_by_no_progress:
        fails.append("draw_by_no_progress mismatch")
    if gh.start_fen != gh2.start_fen:
        fails.append(f"start_fen {gh.start_fen!r} != {gh2.start_fen!r}")
    if gh.reanalyze_count != gh2.reanalyze_count:
        fails.append("reanalyze_count mismatch")
    # policies: compare dense
    if len(gh.policies) != len(gh2.policies):
        fails.append(f"policies len {len(gh.policies)} != {len(gh2.policies)}")
    else:
        maxd = 0.0
        for p1, p2 in zip(gh.policies, gh2.policies):
            a1 = np.zeros(game.action_space_size, dtype=np.float32)
            a1[:len(p1)] = p1
            maxd = max(maxd, float(np.max(np.abs(a1 - p2))))
        if maxd > 1e-6:
            fails.append(f"policies max|d|={maxd}")
    # external_values
    if list(gh.external_values) != list(gh2.external_values):
        if not np.allclose(gh.external_values, gh2.external_values, atol=0, rtol=0):
            fails.append("external_values mismatch")
    return fails


def check_B_gpu_vs_cpu_obs(actions, game_cpu, max_plies=80):
    """Replay the SAME actions through CPU ChessGame and GPU GpuChessGame; compare
    observations plane-by-plane. This is what the .buf reconstruction relies on
    matching the original GPU generation.
    """
    gpu = GpuChessGame()
    gpu.max_plies = 1024
    dev = "cpu"
    gstate = gpu.reset_batch(1, device=dev)

    cstate = game_cpu.reset()

    per_plane_max = np.zeros(ChessGame.num_planes, dtype=np.float64)
    per_plane_cnt = np.zeros(ChessGame.num_planes, dtype=np.int64)
    n_compared = 0
    first_div = None

    # ply 0 obs
    def cmp(ply):
        nonlocal n_compared, first_div
        cpu_obs = game_cpu.to_tensor(cstate).numpy()  # (P,8,8)
        gpu_obs = gpu.to_tensor_batch(gstate)[0].numpy()  # (P,8,8)
        d = np.abs(cpu_obs - gpu_obs)
        for p in range(ChessGame.num_planes):
            m = float(d[p].max())
            per_plane_max[p] = max(per_plane_max[p], m)
            if m > 1e-5:
                per_plane_cnt[p] += 1
        n_compared += 1
        if first_div is None and d.max() > 1e-5:
            bad_planes = [p for p in range(ChessGame.num_planes) if d[p].max() > 1e-5]
            first_div = (ply, bad_planes,
                         {p: (float(cpu_obs[p].max()), float(gpu_obs[p].max()),
                              float(cpu_obs[p].sum()), float(gpu_obs[p].sum())) for p in bad_planes})

    cmp(0)
    for i, a in enumerate(actions[:max_plies]):
        # step cpu
        cstate, _, cdone = game_cpu.step(cstate, int(a))
        # step gpu
        at = torch.tensor([int(a)], dtype=torch.int64, device=dev)
        gstate, _, _, _ = gpu.step_batch_with_legal(gstate, at)
        cmp(i + 1)
        if cdone:
            break

    return per_plane_max, per_plane_cnt, n_compared, first_div


def check_C_make_target_after_save(gh, game, tmp_path):
    """Save game to v3 buffer, reload, compare make_target output."""
    buf = ReplayBuffer(max_size=10)
    buf.save_game(gh)
    buf.save(tmp_path, format_version=3)
    buf2 = ReplayBuffer(max_size=10)
    buf2.load(tmp_path, game=game)
    gh2 = buf2.buffer[0]

    # Deterministic make_target at several positions
    fails = []
    K = 5
    for pos in range(0, len(gh), max(1, len(gh) // 6)):
        t1 = gh.make_target(pos, K, td_steps=10, discount=0.997,
                            action_space_size=game.action_space_size,
                            value_head_type="wdl", history_frames=8)
        t2 = gh2.make_target(pos, K, td_steps=10, discount=0.997,
                            action_space_size=game.action_space_size,
                            value_head_type="wdl", history_frames=8)
        obs1, mask1, act1, val1, rew1, pol1 = t1
        obs2, mask2, act2, val2, rew2, pol2 = t2
        # obs
        for j, (o1, o2) in enumerate(zip(obs1, obs2)):
            md = float(torch.max(torch.abs(o1 - o2)))
            if md > 1e-5:
                fails.append(f"pos{pos} obs[{j}] max|d|={md}")
                break
        if act1 != act2:
            fails.append(f"pos{pos} actions mismatch")
        if mask1 != mask2:
            fails.append(f"pos{pos} mask mismatch")
        v1 = np.array(val1); v2 = np.array(val2)
        if not np.allclose(v1, v2, atol=1e-6):
            fails.append(f"pos{pos} values max|d|={np.max(np.abs(v1-v2))}")
        if not np.allclose(np.array(pol1), np.array(pol2), atol=1e-6):
            fails.append(f"pos{pos} policies mismatch")
    os.remove(tmp_path)
    return fails


def main():
    cfg = get_config("chess_small")
    cfg.device = "cpu"
    game = ChessGame()

    bufdir = "checkpoints/chess/2026_06_19_cold2_pc"
    # Pick a few .buf files: early (big games) and late.
    buf_files = [
        os.path.join(bufdir, "checkpoint_2000.buf"),
        os.path.join(bufdir, "checkpoint_15000.buf"),
        os.path.join(bufdir, "checkpoint_30000.buf"),
    ]

    for bf in buf_files:
        if not os.path.exists(bf):
            print(f"SKIP missing {bf}")
            continue
        print(f"\n===== {os.path.basename(bf)} =====")
        games, header = load_one_buf_game(bf, game, n=3)
        print(f"header={header}")
        for gi, (gh, prio) in enumerate(games):
            print(f"\n--- game {gi}: len={len(gh)} actions={len(gh.actions)} "
                  f"outcome={gh.game_outcome} rep={gh.draw_by_repetition} "
                  f"noprog={gh.draw_by_no_progress} fen={gh.start_fen} "
                  f"ext_vals={len(gh.external_values)} reanalyze={gh.reanalyze_count}")
            # A: roundtrip
            failsA = check_A_roundtrip(gh, game)
            print(f"  [A roundtrip] {'OK' if not failsA else 'FAIL: ' + '; '.join(failsA)}")
            # B: gpu vs cpu obs (THE big one) — only for non-warmstart standard-start games
            if gh.start_fen is None:
                pm, pc, ncmp, fd = check_B_gpu_vs_cpu_obs(gh.actions, ChessGame())
                bad = [p for p in range(ChessGame.num_planes) if pm[p] > 1e-5]
                if not bad:
                    print(f"  [B gpu==cpu obs] OK over {ncmp} plies, all {ChessGame.num_planes} planes match")
                else:
                    print(f"  [B gpu==cpu obs] DIVERGENCE over {ncmp} plies. Bad planes: {bad}")
                    for p in bad:
                        print(f"      plane {p}: max|d|={pm[p]:.4f}, diverged in {pc[p]}/{ncmp} plies")
                    if fd:
                        ply, bp, info = fd
                        print(f"      first divergence at ply {ply}, planes {bp}")
                        for p, (cmx, gmx, csum, gsum) in info.items():
                            print(f"        plane {p}: cpu(max={cmx:.3f},sum={csum:.3f}) gpu(max={gmx:.3f},sum={gsum:.3f})")
            else:
                print(f"  [B gpu==cpu obs] SKIP (start_fen set)")
            # C: make_target after save/reload
            failsC = check_C_make_target_after_save(gh, game, f"/tmp/_probe_{gi}.buf")
            print(f"  [C make_target reload] {'OK' if not failsC else 'FAIL: ' + '; '.join(failsC[:6])}")


if __name__ == "__main__":
    main()
