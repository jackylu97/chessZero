#!/usr/bin/env bash
# A100-80GB run: q-boot (selfplay_q_ratio=0.25) + 800 sims, scaled parallelism.
#   num_parallel_games=1024 (saturation lever; ~40-45 GB VRAM @ max_plies=600, measured-model)
#   num_self_play_games=1024, self_play_interval=660 (preset) -> replay reuse 2.0
#   max_plies=600, replay_buffer_size=5120 (preset; RAM headroom for the N=1024 transfer)
# Resumes the Stockfish-warmstarted chess_small checkpoint at step 15000.
set -euo pipefail
cd /home/user/chessZero

# Reduce fragmentation near the ceiling (also recommended by the earlier OOM msg).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=2026_06_24_a100_qboot_s800_n800
LOG=train_logs/${RUN_ID}.log

exec .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_small --run-id "${RUN_ID}" --device cuda \
  --steps 150000 --eval-interval 2000 --mask-illegal-policy \
  --stockfish-injection-path data/stockfish_injection \
  --stockfish-injection-games 300 --stockfish-injection-interval 256 \
  --self-play-warmup-steps 15000 --warmstart-buffer-size 300 --warmstart-sample-frac 0.4 \
  --material-value-weight 0.5 --material-value-anneal-frac 0.6 \
  --use-material-head --material-head-loss-weight 0.25 \
  --root-terminal-draws --root-terminal-draws-min-repeats 2 --resign-enabled \
  --num-simulations 800 --num-parallel-games 1024 --max-plies 600 \
  --selfplay-q-ratio 0.25 \
  --resume checkpoints/chess/2026_06_23_warmstart_material/checkpoint_15000.pt \
  2>&1 | tee "${LOG}"
