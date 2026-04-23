#!/usr/bin/env bash
# Launch N parallel Stockfish generation workers. Each writes to a worker
# subdir so shard filenames don't collide. Kill with `pkill -f generate_stockfish_games`.
set -euo pipefail

N_WORKERS=${N_WORKERS:-4}
GAMES_PER_WORKER=${GAMES_PER_WORKER:-16000}
OUT_ROOT=${OUT_ROOT:-data/stockfish_injection}
STOCKFISH_HASH=${STOCKFISH_HASH:-64}

mkdir -p "$OUT_ROOT" logs
for i in $(seq 0 $((N_WORKERS - 1))); do
    mkdir -p "$OUT_ROOT/worker_$i"
    .venv/bin/python scripts/generate_stockfish_games.py \
        --out-dir "$OUT_ROOT/worker_$i" \
        --num-games "$GAMES_PER_WORKER" \
        --depth 8 --shard-size 500 \
        --stockfish-threads 1 --stockfish-hash "$STOCKFISH_HASH" \
        --seed $((1000 + i)) \
        > "logs/sf_gen_worker_$i.log" 2>&1 &
    echo "launched worker $i (pid $!, seed $((1000+i)), $GAMES_PER_WORKER games)"
done
wait
echo "all workers finished"
