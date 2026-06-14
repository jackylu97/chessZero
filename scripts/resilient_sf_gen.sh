#!/usr/bin/env bash
# Resilient Stockfish generation. The generator crashes stochastically with a
# native signal (SIGSEGV/SIGILL in stockfish/python-chess) that prints no
# traceback and kills the worker mid-shard. Two defenses:
#   * per-worker auto-restart loop: on crash/hang the worker resumes after its
#     own existing shards (--shard-index-start), with a bumped seed.
#   * small shards (SHARD_SIZE): a crash loses < SHARD_SIZE games, not 500, and
#     shards land within minutes so progress is visible and durable.
#   * per-attempt `timeout`: if a worker hangs on a dead engine pipe, it's
#     killed and resumed rather than stalling forever.
# Fewer workers than the 52-worker batch (less contention => fewer crashes).
set -u
cd /workspace/chessZero

N_WORKERS=${N_WORKERS:-6}                 # per bucket
SHARD_SIZE=${SHARD_SIZE:-50}
TARGET_PER_BUCKET=${TARGET_PER_BUCKET:-64000}
WORKER_INDEX_BASE=${WORKER_INDEX_BASE:-300}
SEED_BASE=${SEED_BASE:-40000}
ATTEMPT_TIMEOUT=${ATTEMPT_TIMEOUT:-2700}  # 45 min/attempt hard cap (hang guard)
STOCKFISH_HASH=${STOCKFISH_HASH:-64}
MULTIPV=10; LABEL_MULTIPV=10; TAU_LABEL=0.10; LABEL_DEPTH=8
OUT_ROOT=data/stockfish_injection
BUCKETS=(8v5 8v6 8v7 8v8)
mkdir -p logs

# Accurate game count for a bucket dir: read each shard's header (compact v2
# carries n_records; legacy is a list). Robust to MIXED shard sizes in the
# pool (old 500-game shards + new small shards), unlike a shards*500 estimate.
count_games() {
  .venv/bin/python - "$1" <<'PY'
import sys, glob, pickle
total = 0
for f in glob.glob(sys.argv[1] + "/**/*.pkl", recursive=True):
    try:
        with open(f, "rb") as fh:
            h = pickle.load(fh)
        if isinstance(h, dict) and h.get("version") == 2:
            total += int(h.get("n_records", 0))
        elif isinstance(h, list):
            total += len(h)
    except Exception:
        pass
print(total)
PY
}

run_worker() {
  local bucket=$1 widx=$2 seed=$3 share=$4
  local outdir="$OUT_ROOT/bucket_$bucket/worker_$widx"
  mkdir -p "$outdir"
  local log="logs/sfres_${bucket}_w${widx}.log"
  local attempt=0
  while :; do
    local have=$(find "$outdir" -name '*.pkl' | wc -l)
    local done_games=$((have * SHARD_SIZE))
    local remaining=$((share - done_games))
    if (( remaining <= 0 )); then
      echo "[$(date +%H:%M:%S) w$widx $bucket] DONE ($done_games games)" >> "$log"; break
    fi
    attempt=$((attempt+1))
    echo "[$(date +%H:%M:%S) w$widx $bucket] attempt $attempt: $have shards / $done_games games, need $remaining" >> "$log"
    timeout "$ATTEMPT_TIMEOUT" .venv/bin/python scripts/generate_stockfish_games.py \
      --out-dir "$outdir" --num-games "$remaining" --bucket "$bucket" \
      --label-depth $LABEL_DEPTH --multipv $MULTIPV --label-multipv $LABEL_MULTIPV \
      --tau-label $TAU_LABEL --shard-size "$SHARD_SIZE" --format-version 2 \
      --shard-index-start "$have" --seed $((seed * 1000 + attempt)) \
      --stockfish-threads 1 --stockfish-hash $STOCKFISH_HASH >> "$log" 2>&1 \
      || echo "[$(date +%H:%M:%S) w$widx $bucket] exited rc=$? — resuming" >> "$log"
    sleep 2
  done
}
export -f run_worker
export OUT_ROOT SHARD_SIZE MULTIPV LABEL_MULTIPV TAU_LABEL LABEL_DEPTH STOCKFISH_HASH ATTEMPT_TIMEOUT

# Compute each bucket's deficit ONCE up front (counting the existing 500-game
# shards). Done before any small shards are written, so the count is clean.
seed=$SEED_BASE
for bi in "${!BUCKETS[@]}"; do
  bucket="${BUCKETS[$bi]}"
  have_games=$(count_games "$OUT_ROOT/bucket_$bucket")
  remaining=$((TARGET_PER_BUCKET - have_games))
  if (( remaining <= 0 )); then echo "bucket $bucket at target ($have_games)"; continue; fi
  share=$(( (remaining + N_WORKERS - 1) / N_WORKERS ))
  echo "bucket $bucket: have $have_games, need $remaining => $share games x $N_WORKERS workers (shard=$SHARD_SIZE)"
  for i in $(seq 0 $((N_WORKERS-1))); do
    widx=$((WORKER_INDEX_BASE + i))
    run_worker "$bucket" "$widx" "$seed" "$share" &
    seed=$((seed+1))
  done
done
echo "launched $((N_WORKERS*4)) resilient workers"
wait
echo "all resilient workers finished"
