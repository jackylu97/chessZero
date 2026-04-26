#!/usr/bin/env bash
# Watchdog: wait for the in-flight Stockfish generation to complete, sanity-check
# the resulting pool, then kick off a supervised chess training run on the new
# data with the Sampled MuZero MCTS fix in place.
set -u

REPO=/home/jacky/code/chessZero
GEN_LOG="$REPO/logs/sf_gen_softmpv_v1.log"
DATA_PATH="data/stockfish_softmpv_v1"
RUN_ID="2026_04_25_0001"
EXPECTED_MIN_SHARDS=60  # 32k games / 500 per shard = 64; tolerate a few short shards

cd "$REPO"

log() { echo "[$(date -Iseconds)] $*"; }

log "post_gen_launcher started; waiting for generation to complete..."

# Poll once per minute. The "all workers finished" line is printed by run_sf_gen.sh
# only after `wait` returns for all background workers.
until grep -q "all workers finished" "$GEN_LOG" 2>/dev/null; do
    sleep 60
done

log "Generation log shows all workers finished."

# Wait briefly for any final shard writes to flush to disk.
sleep 5

# Sanity: shard count.
N_SHARDS=$(find "$DATA_PATH" -name "*.pkl" 2>/dev/null | wc -l)
log "Shard count: $N_SHARDS"
if (( N_SHARDS < EXPECTED_MIN_SHARDS )); then
    log "ABORT: expected at least $EXPECTED_MIN_SHARDS shards, got $N_SHARDS. Not launching training."
    exit 1
fi

# Sanity: total games (sum of game-count printed in worker logs).
log "Per-bucket shard count:"
for b in 8v5 8v6 8v7 8v8; do
    c=$(find "$DATA_PATH/bucket_$b" -name "*.pkl" 2>/dev/null | wc -l)
    log "  bucket_$b: $c shards"
done

# Refuse to overwrite an existing run-id.
if [[ -e "checkpoints/chess/$RUN_ID" ]]; then
    log "ABORT: checkpoints/chess/$RUN_ID already exists. Pick a different run-id."
    exit 1
fi
if [[ -e "runs/chess/$RUN_ID" ]]; then
    log "ABORT: runs/chess/$RUN_ID already exists. Pick a different run-id."
    exit 1
fi

log "Sanity checks passed. Launching supervised training..."
log "  run-id: $RUN_ID"
log "  data:   $DATA_PATH"

# Launch in its own tmux session so this watchdog can exit cleanly.
tmux new-session -d -s chess_train \
    "bash -lc 'scripts/supervise_train.sh --game chess --run-id $RUN_ID --stockfish-injection-path $DATA_PATH'"

log "Training launched. tmux session: chess_train"
log "  attach with: tmux attach -t chess_train"
log "  drain via:   touch runs/chess/$RUN_ID/STOP"
log "post_gen_launcher exiting."
