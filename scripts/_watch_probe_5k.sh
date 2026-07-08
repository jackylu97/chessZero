#!/usr/bin/env bash
# Wait for scale_pc checkpoint_5000 to land, then run the within-position
# sibling value-spread probe (the broken quantity, Stockfish-free) on the
# scaled run AND the cold2_pc baseline at matched step 5000, and dump the
# repr-probe (r2_out/eff_rank) trajectory. Writes a report and exits
# (re-invoking the parent agent).
set -u
cd /workspace/chessZero
RUN=${1:-2026_06_20_scale_gen}
CKDIR=checkpoints/chess/$RUN
PT=$CKDIR/checkpoint_5000.pt
BUF=$CKDIR/checkpoint_5000.buf
REPORT=logs/probe_scale_5k_report.txt
PY=.venv/bin/python

# Poll up to ~7h (420 * 60s) for both files.
for i in $(seq 1 420); do
  [[ -f "$PT" && -f "$BUF" ]] && break
  sleep 60
done

{
  echo "===================================================================="
  echo "5K PROBE REPORT — $RUN  ($(date -Iseconds))"
  echo "===================================================================="
  if [[ ! -f "$PT" || ! -f "$BUF" ]]; then
    echo "TIMEOUT: checkpoint_5000 (.pt/.buf) never appeared after ~7h."
    echo "Current checkpoints:"; ls -v "$CKDIR" 2>/dev/null | grep -E '\.pt$' | tail -8
    exit 0
  fi

  echo; echo "### Within-position sibling value SPREAD (the broken quantity)"
  echo "### cold2_pc baseline reference: sib-Q std ~0.0076, across-pos root-V std ~0.20, ratio ~0.04"
  echo
  echo "--- SCALED run  $RUN @ step 5000 ---"
  PROBE_CKPT="$PT" PROBE_BUF="$BUF" $PY scripts/probe_sibling_value_spread.py 2>/dev/null

  echo; echo "--- BASELINE  cold2_pc @ step 5000 (matched) ---"
  PROBE_CKPT=checkpoints/chess/2026_06_19_cold2_pc/checkpoint_5000.pt \
  PROBE_BUF=checkpoints/chess/2026_06_19_cold2_pc/checkpoint_5000.buf \
    $PY scripts/probe_sibling_value_spread.py 2>/dev/null

  echo; echo "### Repr-probe trajectory (r2_out / eff_rank) — SCALED run so far"
  echo "### cold2_pc collapsed: 4k r2=0.59/rank245 -> 6k 0.10/240 -> 12k 0.11/199 -> 20k 0.10/152"
  grep -E "repr probe" logs/${RUN}.log 2>/dev/null | tail -12

  echo; echo "### Self-play draw rate / game length (SCALED run)"
  grep -iE "draw|game.len|avg.len|decisive" logs/${RUN}.log 2>/dev/null | tail -8
  echo "===================================================================="
} > "$REPORT" 2>&1

echo "PROBE COMPLETE — wrote $REPORT"
cat "$REPORT"
