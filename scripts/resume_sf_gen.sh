#!/usr/bin/env bash
# Resume the asymmetric Stockfish warmstart pool up to TARGET_PER_BUCKET games
# per bucket, WITHOUT clobbering or duplicating existing shards.
#
# How it avoids collisions:
#   * New workers write to bucket_$b/worker_$((WORKER_INDEX_BASE + i)), disjoint
#     from the original worker_0..3 dirs. Training globs *.pkl recursively, so
#     all workers (old + new) are picked up as one pool.
#   * New seeds start at SEED_BASE (far from the original 1000-range), so the
#     RNG streams are disjoint and new games are diverse, not re-rolls.
#   * Shards are written in the EFFICIENT compact v2 format (--format-version 2,
#     the generator default): ~23 KB/game, obs replayed at load. This is what
#     blew the disk last time when it was the 3.7 MB/game legacy format.
#
# Per bucket: remaining = TARGET_PER_BUCKET - existing_games, split evenly over
# N_WORKERS. Buckets already at/over target are skipped.
#
# Kill with: pkill -f generate_stockfish_games
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-data/stockfish_injection}
TARGET_PER_BUCKET=${TARGET_PER_BUCKET:-64000}   # original = 16000 games x 4 workers
N_WORKERS=${N_WORKERS:-16}                       # new workers per bucket
WORKER_INDEX_BASE=${WORKER_INDEX_BASE:-100}      # disjoint from original 0..3
SEED_BASE=${SEED_BASE:-20000}                    # disjoint from original 1000-range
SHARD_SIZE=${SHARD_SIZE:-500}
STOCKFISH_HASH=${STOCKFISH_HASH:-64}
MULTIPV=${MULTIPV:-10}
LABEL_MULTIPV=${LABEL_MULTIPV:-10}
TAU_LABEL=${TAU_LABEL:-0.10}
LABEL_DEPTH=${LABEL_DEPTH:-8}
BUCKETS=(8v5 8v6 8v7 8v8)

mkdir -p logs
echo "Resume target=$TARGET_PER_BUCKET/bucket, $N_WORKERS new workers/bucket, compact v2"

seed=$SEED_BASE
for bi in "${!BUCKETS[@]}"; do
    bucket="${BUCKETS[$bi]}"
    existing_shards=$(find "$OUT_ROOT/bucket_$bucket" -name '*.pkl' 2>/dev/null | wc -l)
    existing_games=$((existing_shards * SHARD_SIZE))
    remaining=$((TARGET_PER_BUCKET - existing_games))
    if (( remaining <= 0 )); then
        echo "bucket $bucket: $existing_games >= target, skipping"
        continue
    fi
    per_worker=$(( (remaining + N_WORKERS - 1) / N_WORKERS ))
    echo "bucket $bucket: have $existing_games, need $remaining → $per_worker games x $N_WORKERS workers"
    for i in $(seq 0 $((N_WORKERS - 1))); do
        widx=$((WORKER_INDEX_BASE + i))
        out_dir="$OUT_ROOT/bucket_$bucket/worker_$widx"
        mkdir -p "$out_dir"
        .venv/bin/python scripts/generate_stockfish_games.py \
            --out-dir "$out_dir" \
            --num-games "$per_worker" \
            --bucket "$bucket" \
            --label-depth "$LABEL_DEPTH" \
            --multipv "$MULTIPV" \
            --label-multipv "$LABEL_MULTIPV" \
            --tau-label "$TAU_LABEL" \
            --shard-size "$SHARD_SIZE" \
            --format-version 2 \
            --stockfish-threads 1 --stockfish-hash "$STOCKFISH_HASH" \
            --seed "$seed" \
            > "logs/sf_resume_${bucket}_worker_$widx.log" 2>&1 &
        seed=$((seed + 1))
    done
done
echo "launched all resume workers; waiting..."
wait
echo "all resume workers finished"
