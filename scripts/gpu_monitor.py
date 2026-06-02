"""Decoupled GPU telemetry -> TensorBoard.

Polls ``nvidia-smi`` every --interval seconds and writes ``gpu/*`` scalars to a
separate event file under ``<logdir>/gpu_monitor`` so TensorBoard merges it with
the training run's curves. Runs as its own process (own tmux pane), so it:
  - captures GPU utilization DURING self-play (the trainer's boundary snapshots
    only sample GPU memory at self-play edges, and never log saturation),
  - survives train.py crashes/restarts (independent lifetime),
  - adds zero GPU compute (nvidia-smi query only).

X-axis is wall-clock seconds since monitor start (decoupled from train steps).

Usage:
  .venv/bin/python scripts/gpu_monitor.py --logdir runs/chess/<run_id> --interval 5
"""
import argparse
import os
import subprocess
import time

from torch.utils.tensorboard import SummaryWriter

# (tag, nvidia-smi field). utilization.gpu is the SM saturation %; utilization.memory
# is the % of time the memory bus was read/written (a copy-bound indicator, distinct
# from how-full memory is — that's memory.used / memory.total).
FIELDS = [
    ("util_pct", "utilization.gpu"),
    ("mem_bus_pct", "utilization.memory"),
    ("mem_used_mb", "memory.used"),
    ("mem_total_mb", "memory.total"),
    ("power_w", "power.draw"),
    ("temp_c", "temperature.gpu"),
    ("sm_clock_mhz", "clocks.sm"),
]


def sample():
    query = ",".join(f for _, f in FIELDS)
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15,
    )
    rows = []
    for line in out.stdout.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        # Robust per-field parse: [N/A] / "" -> None (skipped), else float.
        parsed = {}
        for (tag, _), cell in zip(FIELDS, cells):
            try:
                parsed[tag] = float(cell)
            except (ValueError, TypeError):
                parsed[tag] = None
        rows.append(parsed)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="run dir, e.g. runs/chess/<run_id>")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    writer = SummaryWriter(log_dir=os.path.join(args.logdir, "gpu_monitor"))
    print(f"gpu_monitor: writing gpu/* to {args.logdir}/gpu_monitor every {args.interval}s")
    t0 = time.monotonic()
    n = 0
    while True:
        step = int(time.monotonic() - t0)
        try:
            rows = sample()
        except Exception as e:
            print(f"gpu_monitor: sample failed ({e}); retrying")
            time.sleep(args.interval)
            continue
        for i, parsed in enumerate(rows):
            pfx = f"gpu{i}" if len(rows) > 1 else "gpu"
            for tag, val in parsed.items():
                if val is None:
                    continue
                writer.add_scalar(f"{pfx}/{tag}", val, step)
            mu, mt = parsed.get("mem_used_mb"), parsed.get("mem_total_mb")
            if mu is not None and mt:
                writer.add_scalar(f"{pfx}/mem_used_pct", 100.0 * mu / mt, step)
        writer.flush()
        n += 1
        if n % 12 == 0:  # heartbeat roughly every minute at 5s
            u = rows[0].get("util_pct") if rows else None
            print(f"gpu_monitor: t={step}s util={u}% (sample #{n})")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
