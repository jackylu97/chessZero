#!/usr/bin/env bash
# Launch N parallel Stockfish generation workers. Two modes:
#
#   BUCKET_SWEEP=0 (default)  — symmetric depth-X self-play pool. Uses --depth (default 8).
#                                Workers write to $OUT_ROOT/worker_$i.
#
#   BUCKET_SWEEP=1            — asymmetric-teacher pool across 4 buckets (8v5, 8v6, 8v7,
#                                8v8). Labels at depth 8 (depth-8 top move = policy target
#                                + depth-8 STM eval = external_values); plays at the
#                                bucket's depth pair with MultiPV=$MULTIPV softmax
#                                sampling. Colors alternate 50/50 per game within each
#                                bucket. Shards go to $OUT_ROOT/bucket_$b/worker_$i so
#                                training's recursive glob picks them all up as one pool.
#                                GAMES_PER_WORKER is per-bucket-per-worker.
#
# Kill with `pkill -f generate_stockfish_games`.
set -euo pipefail

N_WORKERS=${N_WORKERS:-4}
GAMES_PER_WORKER=${GAMES_PER_WORKER:-16000}
OUT_ROOT=${OUT_ROOT:-data/stockfish_injection}
STOCKFISH_HASH=${STOCKFISH_HASH:-64}
BUCKET_SWEEP=${BUCKET_SWEEP:-0}
MULTIPV=${MULTIPV:-5}          # K for play-side softmax. 5 pragmatic; 10 = ~2× cost.
LABEL_DEPTH=${LABEL_DEPTH:-8}  # depth for policy/value labels under asymmetric mode.
DEPTH=${DEPTH:-8}              # symmetric-mode play depth (ignored under BUCKET_SWEEP=1).

mkdir -p "$OUT_ROOT" logs

if [[ "$BUCKET_SWEEP" == "1" ]]; then
    BUCKETS=(8v5 8v6 8v7 8v8)
    echo "Bucket sweep over ${BUCKETS[*]} ($N_WORKERS workers/bucket, $GAMES_PER_WORKER games each)"
    # Seed offsets keep workers' RNGs disjoint across the whole sweep so the same
    # opening is unlikely to repeat in different worker/bucket pairs.
    seed_base=1000
    for bi in "${!BUCKETS[@]}"; do
        bucket="${BUCKETS[$bi]}"
        for i in $(seq 0 $((N_WORKERS - 1))); do
            out_dir="$OUT_ROOT/bucket_$bucket/worker_$i"
            mkdir -p "$out_dir"
            seed=$((seed_base + bi * N_WORKERS + i))
            .venv/bin/python scripts/generate_stockfish_games.py \
                --out-dir "$out_dir" \
                --num-games "$GAMES_PER_WORKER" \
                --bucket "$bucket" \
                --label-depth "$LABEL_DEPTH" \
                --multipv "$MULTIPV" \
                --shard-size 500 \
                --stockfish-threads 1 --stockfish-hash "$STOCKFISH_HASH" \
                --seed "$seed" \
                > "logs/sf_gen_${bucket}_worker_$i.log" 2>&1 &
            echo "launched $bucket worker $i (pid $!, seed $seed, $GAMES_PER_WORKER games)"
        done
    done
else
    for i in $(seq 0 $((N_WORKERS - 1))); do
        mkdir -p "$OUT_ROOT/worker_$i"
        .venv/bin/python scripts/generate_stockfish_games.py \
            --out-dir "$OUT_ROOT/worker_$i" \
            --num-games "$GAMES_PER_WORKER" \
            --depth "$DEPTH" --shard-size 500 \
            --stockfish-threads 1 --stockfish-hash "$STOCKFISH_HASH" \
            --seed $((1000 + i)) \
            > "logs/sf_gen_worker_$i.log" 2>&1 &
        echo "launched worker $i (pid $!, seed $((1000+i)), $GAMES_PER_WORKER games)"
    done
fi
wait
echo "all workers finished"
