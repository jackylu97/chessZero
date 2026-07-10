"""Background checkpoint+buffer janitor for long runs.

Keeps every KEEP_EVERY-step checkpoint (the cadence prod_probes reads) plus the
KEEP_RECENT most-recent step, and deletes the rest -- BOTH the .pt AND the
sibling .buf (replay-buffer snapshot). Loops every INTERVAL seconds; ONESHOT=1
prunes once and exits (one-time cleanup of a finished run). Only ever touches
files matching checkpoint_<int>.{pt,buf} in the given dir.

Usage:  .venv/bin/python scripts/prune_checkpoints.py <checkpoint_dir>
Env:    KEEP_EVERY (default 10000), KEEP_RECENT (1), INTERVAL (600s),
        ONESHOT (0), DRYRUN (0)
"""
import glob, os, re, sys, time

d = sys.argv[1]
KEEP_EVERY = int(os.environ.get("KEEP_EVERY", 10000))
KEEP_RECENT = int(os.environ.get("KEEP_RECENT", 1))
INTERVAL = int(os.environ.get("INTERVAL", 600))
ONESHOT = os.environ.get("ONESHOT", "0") == "1"
DRYRUN = os.environ.get("DRYRUN", "0") == "1"


def step_of(f):
    m = re.search(r"checkpoint_(\d+)\.(pt|buf)$", os.path.basename(f))
    return int(m.group(1)) if m else -1


def prune_once():
    files = [f for f in glob.glob(os.path.join(d, "checkpoint_*")) if step_of(f) >= 0]
    steps = sorted({step_of(f) for f in files})
    if not steps:
        return 0
    recent = set(steps[-KEEP_RECENT:]) if KEEP_RECENT > 0 else set()
    keep = {s for s in steps if s % KEEP_EVERY == 0} | recent
    victims = [f for f in files if step_of(f) not in keep]
    if DRYRUN:
        gb = sum(os.path.getsize(f) for f in victims) / 1e9
        print(f"DRYRUN: keep steps {sorted(keep)}; would delete {len(victims)} "
              f"files (~{gb:.1f}GB)", flush=True)
        return 0
    removed = 0
    for f in victims:
        try:
            os.remove(f); removed += 1
        except OSError:
            pass
    if removed:
        print(f"{time.strftime('%H:%M:%S')} kept {len(keep)} steps (.pt+.buf) "
              f"removed {removed} files (range {steps[0]}..{steps[-1]})", flush=True)
    return removed


print(f"janitor: dir={d} keep_every={KEEP_EVERY} keep_recent={KEEP_RECENT} "
      f"oneshot={ONESHOT} dryrun={DRYRUN} interval={INTERVAL}s", flush=True)
if ONESHOT or DRYRUN:
    prune_once()
else:
    while True:
        prune_once()
        time.sleep(INTERVAL)
