"""One-off: compare retention_m7 vs anchor04 at matched phase-2 steps.
Called by the phase-2 watcher. Read-only TB scrape."""
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {"retention_m7": "2026_06_16_retention_m7",
        "anchor04": "2026_06_16_maskpolicy_anchor04"}
EAS = {}
for k, v in RUNS.items():
    fs = glob.glob(f"runs/chess/{v}/*.tfevents*")
    if fs:
        ea = EventAccumulator(fs[0], size_guidance={"scalars": 0}); ea.Reload(); EAS[k] = ea

def at(ea, t, step):
    try:
        s = [(x.step, x.value) for x in ea.Scalars(t)]
    except Exception:
        return None
    if not s:
        return None
    c = [p for p in s if p[0] <= step]
    return (c[-1] if c else s[0])[1]

for k, ea in EAS.items():
    try:
        print(f"{k}: latest step {max(x.step for x in ea.Scalars('loss/total_loss'))}")
    except Exception:
        pass
print("\n=== retention_m7 vs anchor04 (phase 2, matched steps) ===")
for t in ["self_play/buffer_decisive_frac", "value/target_std", "self_play/draw_rate",
          "self_play/p1_win_rate", "policy/entropy_pred", "policy/illegal_mass"]:
    print(t)
    for step in [16000, 18000, 20000, 24000]:
        row = []
        for k, ea in EAS.items():
            v = at(ea, t, step)
            row.append(f"{k}={v:.3f}" if v is not None else f"{k}=--")
        print(f"  {step//1000}k: " + "  ".join(row))
print("\nNB: buffer_decisive_frac only logged for retention_m7 (added with the multiplier); "
      "anchor04 predates it. retention_m7 climbing toward ~0.27 = M=7 working.")
