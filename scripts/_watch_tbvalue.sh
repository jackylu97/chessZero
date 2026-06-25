#!/bin/bash
# Watch the tb_value run and dashboard each NEW checkpoint (CPU, so it doesn't
# contend with the GPU training). Appends to logs/dashboard_tbvalue.log so the
# value-DTZ-corr trend is visible over training. Poll every 10 min.
cd /workspace/chessZero
CKPT_DIR=checkpoints/chess/2026_06_25_tb_value
LOG=logs/dashboard_tbvalue.log
EVERY_STEPS=${1:-5000}   # only dashboard every N steps (default 5k) to keep it cheap
mkdir -p logs
echo "watching $CKPT_DIR (dashboard every ${EVERY_STEPS} steps) -> $LOG"
seen=""
while true; do
  latest=$(ls -t "$CKPT_DIR"/checkpoint_*.pt 2>/dev/null | head -1)
  if [ -n "$latest" ]; then
    step=$(echo "$latest" | grep -oE '[0-9]+' | tail -1)
    if [ -n "$step" ] && [ "$step" != "$seen" ] && [ $((step % EVERY_STEPS)) -eq 0 ]; then
      echo "=== dashboard @ step $step  $(date -u +%H:%M:%S) ===" >> "$LOG"
      PYTHONPATH=. .venv/bin/python scripts/eval_relabel_dashboard.py \
        --checkpoint "$latest" --device cpu >> "$LOG" 2>&1
      seen="$step"
      echo "  dashboarded step $step"
    fi
  fi
  sleep 600
done
