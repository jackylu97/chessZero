"""Bridge a train_tb_endgame.py stdout log into TensorBoard scalars (backfill + follow).

train_tb_endgame.py prints metrics to stdout (no native TB). This tails that log,
parses the `step ...` lines, and writes scalars to a TB event dir so the run shows up
in TensorBoard. Backfills existing history on start, then follows live.

Run: PYTHONPATH=. .venv/bin/python scripts/log_to_tb.py <logfile> <tag>
e.g. scripts/log_to_tb.py scaled_mllin24.log mllin24
"""
import sys, os, re, time
from torch.utils.tensorboard import SummaryWriter

LOG = sys.argv[1] if len(sys.argv) > 1 else "scaled_mllin24.log"
TAG = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(LOG))[0]
OUT = f"runs/chess/{TAG}"
w = SummaryWriter(OUT)
print(f"log_to_tb: {LOG} -> {OUT}", flush=True)

STEP = re.compile(r"^step\s+(\d+)")
def grab(pat, line):
    m = re.search(pat, line)
    return float(m.group(1)) if m else None

def emit(line):
    sm = STEP.match(line)
    if not sm:
        return
    step = int(sm.group(1))
    pairs = {
        "train/loss": grab(r"loss\s+([\d.]+)", line),
        "eval/value_acc": grab(r"value_acc\s+([\d.]+)", line),
        "eval/policy_acc": grab(r"policy_acc\s+([\d.]+)", line),
        "ml/won": grab(r"ml_won\s+([\d.]+)", line),
        "ml/draw": grab(r"ml_draw\s+([\d.]+)", line),
        "mcts/conversion": grab(r"CONV\s+([\d.]+)", line),
        "mcts/stalemate_draw": grab(r"stalemate-DRAW\s+([\d.]+)", line),
        "mcts/cap": grab(r"cap\s+([\d.]+)", line),
    }
    for k, v in pairs.items():
        if v is not None:
            w.add_scalar(k, v, step)
    w.flush()

# backfill, then follow
with open(LOG) as f:
    for line in f:
        emit(line)
    while True:
        line = f.readline()
        if line:
            emit(line)
        else:
            time.sleep(2.0)
