#!/usr/bin/env bash
# Wait for the material run's checkpoint_5000 to land, then run the
# within-position sibling value↔Stockfish ranking probe (the broken quantity)
# and write a report. Re-invokes the parent agent on completion.
set -u
cd /workspace/chessZero
RUN=${1:-2026_06_23_cold_material_anneal}
PT=checkpoints/chess/$RUN/checkpoint_5000.pt
SF=tools/stockfish/stockfish
REPORT=logs/probe_5k_${RUN}.txt
PY=.venv/bin/python

# Poll up to ~8h for the checkpoint.
for i in $(seq 1 480); do
  [[ -f "$PT" ]] && break
  sleep 60
done

{
  echo "===================================================================="
  echo "5K SIBLING-RANKING PROBE — $RUN  ($(date -Iseconds))"
  echo "===================================================================="
  if [[ ! -f "$PT" ]]; then
    echo "TIMEOUT: $PT never appeared after ~8h."
    ls -v "checkpoints/chess/$RUN" 2>/dev/null | grep -E '\.pt$' | tail -8
    exit 0
  fi
  echo "checkpoint: $PT"
  echo
  echo "### probe_sibling_ranking (within-position value vs Stockfish; the broken quantity)"
  PYTHONPATH=/workspace/chessZero $PY scripts/probe_sibling_ranking.py \
    --checkpoint "$PT" --game chess_small --stockfish "$SF" \
    --positions 30 --sims 60 --sf-depth 12 2>&1
  echo
  echo "### self-play draw/decisive trend (TB-side, from the train log)"
  grep -iE "draw_rate|decisive|avg.len|p1_win" "logs/${RUN}.log" 2>/dev/null | tail -10
  echo "===================================================================="
} > "$REPORT" 2>&1

echo "5K PROBE COMPLETE — wrote $REPORT"
cat "$REPORT"
