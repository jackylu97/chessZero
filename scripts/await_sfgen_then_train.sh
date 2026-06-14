#!/usr/bin/env bash
# Wait for the Stockfish resume generation (resume_sf_gen.sh workers) to finish,
# then cleanly stop the current qratio_clean run and launch the fresh 22-plane,
# two-pool-anchor run. Self-contained: the stop+launch happen inside this script
# so they occur even if the controlling session is gone.
set -u
cd /workspace/chessZero

OLD_RUN=2026_06_14_qratio_clean
NEW_RUN=2026_06_14_22plane_anchor
MIN_GAMES=100000   # sanity floor: abort launch if the pool is smaller than this

echo "[await] $(date -Iseconds) waiting for sfgen launcher to finish..."
# Wait on the resilient launcher process, NOT the worker procs: workers
# auto-restart across crashes, so the worker set briefly empties between
# attempts and pgrep on them would race. The launcher exits only when its
# `wait` returns (all workers reached their share).
while pgrep -f resilient_sf_gen.sh >/dev/null 2>&1; do
  sleep 60
done
sleep 5
echo "[await] sfgen launcher exited. master log tail:"
tail -3 logs/sf_resilient_master.log 2>/dev/null

# Sanity gate on pool size (shards x 500 games).
shards=$(find data/stockfish_injection -name '*.pkl' | wc -l)
games=$((shards * 500))
echo "[await] pool ~= $games games across $shards shards"
if (( games < MIN_GAMES )); then
  echo "[await] ABORT: pool < $MIN_GAMES; not launching. Investigate sfgen logs."
  exit 1
fi

# --- Stop the old run (no GPU sharing) -------------------------------------
echo "[await] stopping old run $OLD_RUN ..."
mkdir -p runs/chess/$OLD_RUN
touch runs/chess/$OLD_RUN/STOP                 # supervisor exits 0 at next loop top
pkill -TERM -f "scripts/train.py --game chess --run-id $OLD_RUN" 2>/dev/null || true
sleep 30                                        # let it flush + free CUDA + supervisor see rc=143/STOP
pkill -TERM -f "supervise_train.sh --game chess --run-id $OLD_RUN" 2>/dev/null || true
sleep 5
if pgrep -f "run-id $OLD_RUN" >/dev/null 2>&1; then
  echo "[await] old run still alive, SIGKILL"
  pkill -KILL -f "run-id $OLD_RUN" 2>/dev/null || true
  sleep 5
fi
echo "[await] GPU after stop:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null

# --- Launch the new run -----------------------------------------------------
# Mirrors the qratio_clean launch (which runs GPU self-play via the chess
# preset) and adds --warmstart-buffer-size 300 to enable the persistent
# two-pool anchor. Fresh run-id => cold start on the new 22-plane encoding.
echo "[await] launching new run $NEW_RUN ..."
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train "./scripts/supervise_train.sh --game chess --run-id $NEW_RUN \
  --stockfish-injection-path data/stockfish_injection \
  --stockfish-injection-games 1024 --stockfish-injection-interval 256 \
  --warmstart-sample-frac 0.4 --warmstart-buffer-size 300 \
  --train-log logs/train_${NEW_RUN}.log"
sleep 20
echo "[await] new run procs:"
pgrep -af "scripts/train.py --game chess --run-id $NEW_RUN" || echo "[await] WARNING: new train proc not found yet (check logs/train_${NEW_RUN}.log)"
echo "[await] $(date -Iseconds) DONE"
