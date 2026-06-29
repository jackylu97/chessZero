"""Trace actual self-play games + value targets from checkpoint_6000.buf,
and run the value head on real positions. Distinguishes
'value head genuinely cannot tell' from 'value right but downstream draws'.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer
from scripts.eval_checkpoint_health import build_network

CKPT = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
BUF = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"


def main():
    torch.serialization.add_safe_globals([MuZeroConfig])
    cfg = get_config("chess_small"); cfg.device = "cpu"
    game = ChessGame()
    HF = cfg.history_frames

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = build_network(ckpt, game, cfg, "cpu")
    print("trained steps:", ckpt.get("step", ckpt.get("training_step", "?")))
    print("value_head_type:", cfg.value_head_type, "draw_score:", cfg.draw_score,
          "history_frames:", HF, "num_planes:", game.num_planes)

    # Load buffer
    rb = ReplayBuffer(max_size=100000)
    rb.load(BUF, game=game)
    games = rb.buffer
    print(f"\nloaded {len(games)} games from buf")

    # --- Game-level stats ---
    lengths = np.array([len(g) for g in games])
    outcomes = np.array([g.game_outcome for g in games])
    rep = np.array([bool(getattr(g, "draw_by_repetition", False)) for g in games])
    nop = np.array([bool(getattr(g, "draw_by_no_progress", False)) for g in games])
    print(f"len: min={lengths.min()} max={lengths.max()} mean={lengths.mean():.1f} median={np.median(lengths):.0f}")
    print(f"outcome dist: draws(|z|<.5)={np.mean(np.abs(outcomes)<0.5)*100:.1f}%  "
          f"decisive={np.mean(np.abs(outcomes)>=0.5)*100:.1f}%")
    print(f"draw_by_repetition flag: {rep.mean()*100:.1f}%   draw_by_no_progress: {nop.mean()*100:.1f}%")
    ext = sum(1 for g in games if getattr(g, "external_values", None))
    print(f"games with external_values (warmstart): {ext}")

    # --- root_values distribution (the network's OWN value estimate during self-play) ---
    all_rv = np.concatenate([np.asarray(g.root_values, dtype=np.float32) for g in games if len(g.root_values)])
    print(f"\nroot_values: n={len(all_rv)} mean={all_rv.mean():.3f} std={all_rv.std():.3f} "
          f"min={all_rv.min():.3f} max={all_rv.max():.3f}")
    print(f"  |rv|<0.1: {np.mean(np.abs(all_rv)<0.1)*100:.1f}%   |rv|>0.5: {np.mean(np.abs(all_rv)>0.5)*100:.1f}%")

    # --- Run the value head FRESH on real stored positions (with history stacks) ---
    # Sample positions from the FRONT (opening), MIDDLE, and END of draw games.
    def stack_obs(g, idx):
        return g._stack_history(idx, HF).unsqueeze(0)

    @torch.no_grad()
    def val_and_wdl(obs):
        hidden, _pl, value = net.initial_inference(obs)
        v = float(value.reshape(-1)[0])
        logits = net.prediction.value_head(hidden)
        probs = torch.softmax(logits, dim=-1).reshape(-1).tolist()  # (W, D, L)
        return v, probs

    # Pick a long draw game and look at value across plies
    draw_games = [g for g in games if abs(g.game_outcome) < 0.5 and len(g) > 30]
    print(f"\n# long draw games (>30 plies): {len(draw_games)}")
    if draw_games:
        g = max(draw_games, key=len)
        print(f"\n=== tracing value head across a {len(g)}-ply draw game (outcome={g.game_outcome}) ===")
        print(f"  draw_by_rep={getattr(g,'draw_by_repetition',None)} draw_by_nop={getattr(g,'draw_by_no_progress',None)}")
        idxs = list(range(0, len(g), max(1, len(g) // 14)))
        print(f"{'ply':>4} {'net_V':>8} {'root_V':>8} {'P(W)':>6} {'P(D)':>6} {'P(L)':>6} {'rep2':>5} {'rep3':>5} {'np':>6}")
        for idx in idxs:
            obs = stack_obs(g, idx)
            v, probs = val_and_wdl(obs)
            rv = g.root_values[idx] if idx < len(g.root_values) else float("nan")
            frame0 = obs[0, :game.num_planes]
            rep2, rep3, npl = float(frame0[19,0,0]), float(frame0[20,0,0]), float(frame0[21,0,0])
            print(f"{idx:>4} {v:>8.3f} {rv:>8.3f} {probs[0]:>6.3f} {probs[1]:>6.3f} {probs[2]:>6.3f} {rep2:>5.1f} {rep3:>5.1f} {npl:>6.3f}")


if __name__ == "__main__":
    main()
