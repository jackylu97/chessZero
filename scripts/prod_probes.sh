#!/bin/bash
# Standing instrument loop for the production run: at every NEW 10k-step
# checkpoint, run the reward-precision probe + the conversion diag under
# TRAINING CONDITIONS (Gumbel@400) and append to production_probes.log.
# Run in its own tmux: bash scripts/prod_probes.sh <run-id>
RUN="${1:?usage: prod_probes.sh <run-id>}"
cd "$(dirname "$0")/.."
LOG=production_probes.log
SEEN=""
while true; do
  for CK in checkpoints/chess/$RUN/checkpoint_*0000.pt; do
    [ -f "$CK" ] || continue
    case "$SEEN" in *"$CK"*) continue;; esac
    STEP=$(basename "$CK" | sed 's/checkpoint_//; s/\.pt//')
    if [ $((STEP % 10000)) -ne 0 ]; then SEEN="$SEEN $CK"; continue; fi
    echo "===== $CK $(date)" >> $LOG
    echo "-- reward precision:" >> $LOG
    CFG=chess_hybrid_xl REWARD_PLANES=8 POLICY_HEAD=from_to PYTHONPATH=. \
      .venv/bin/python -u scripts/probe_reward_precision.py "$CK" >> $LOG 2>&1
    echo "-- conversion diag (80-ply, N=250):" >> $LOG
    CKPT="$CK" CFG=chess_hybrid_xl USE_ATTENTION=1 USE_SMOLGEN=1 USE_DYN_ATTENTION=1 \
      USE_PRED_ATTENTION=1 MLH_PLANES=8 MLH_BLOCKS=0 USE_GUMBEL=1 SIMS=400 \
      REWARD_PLANES=8 POLICY_HEAD=from_to N_WON=250 SKIP_TERM=1 PYTHONPATH=. \
      .venv/bin/python -u scripts/diag_perconfig_mcts.py 2>&1 | \
      grep -E "CONVERTED|mean plies|K.vK|OutOfMemory" >> $LOG
    SEEN="$SEEN $CK"
  done
  sleep 1800
done
