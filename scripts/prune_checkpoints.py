"""Background checkpoint janitor for long runs.

Keeps every KEEP_EVERY-step checkpoint (the cadence prod_probes reads) plus the
KEEP_RECENT most recent, deletes the rest. Loops every INTERVAL seconds. Only
ever touches files matching checkpoint_<int>.pt in the given dir.

Usage:  .venv/bin/python scripts/prune_checkpoints.py <checkpoint_dir>
Env:    KEEP_EVERY (default 10000), KEEP_RECENT (5), INTERVAL (600s)
"""
import glob, os, re, sys, time

d = sys.argv[1]
KEEP_EVERY = int(os.environ.get("KEEP_EVERY", 10000))
KEEP_RECENT = int(os.environ.get("KEEP_RECENT", 5))
INTERVAL = int(os.environ.get("INTERVAL", 600))

def step(f):
    m = re.search(r"checkpoint_(\d+)\.pt$", os.path.basename(f))
    return int(m.group(1)) if m else -1

print(f"janitor: dir={d} keep_every={KEEP_EVERY} keep_recent={KEEP_RECENT} "
      f"interval={INTERVAL}s", flush=True)
while True:
    files = [f for f in glob.glob(os.path.join(d, "checkpoint_*.pt")) if step(f) >= 0]
    steps = sorted(step(f) for f in files)
    if steps:
        recent = set(steps[-KEEP_RECENT:])
        keep = {s for s in steps if s % KEEP_EVERY == 0} | recent
        removed = 0
        for f in files:
            if step(f) not in keep:
                try:
                    os.remove(f); removed += 1
                except OSError:
                    pass
        if removed:
            print(f"{time.strftime('%H:%M:%S')} kept {len(keep)} removed {removed} "
                  f"(range {steps[0]}..{steps[-1]})", flush=True)
    time.sleep(INTERVAL)
