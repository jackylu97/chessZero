"""Long-run trajectory dump for retention_m7 (called by the long-run watcher)."""
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

fs = glob.glob("runs/chess/2026_06_16_retention_m7/*.tfevents*")
ea = EventAccumulator(fs[0], size_guidance={"scalars": 0}); ea.Reload()
latest = max(x.step for x in ea.Scalars("loss/total_loss"))
print(f"retention_m7 — latest step {latest}  (LR drops at 75k & 112.5k; ends 150k)")

def traj(t, pts=12):
    try:
        s = [(x.step, x.value) for x in ea.Scalars(t)]
    except Exception:
        return "n/a"
    if not s:
        return "n/a"
    n = len(s); idx = sorted(set(int(i * (n - 1) / (pts - 1)) for i in range(pts)))
    return "  ".join(f"{s[i][0]//1000}k:{s[i][1]:.2f}" for i in idx)

for t in ["self_play/draw_rate", "self_play/p1_win_rate", "eval/win_rate_vs_random",
          "value/target_std", "self_play/buffer_decisive_frac", "policy/entropy_pred",
          "policy/illegal_mass", "train/lr"]:
    print(f"  {t:30} {traj(t)}")
