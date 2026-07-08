#!/bin/bash
# Launch the qboot_s800 run (selfplay_q_ratio=0.25, 800 sims, 512 games, 128 parallel),
# resumed from the 15k warmstart checkpoint. Output goes to the tmux pane AND the log.
cd /workspace/chessZero
.venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_small \
  --run-id 2026_06_24_warmstart_qboot_s800 \
  --device cuda \
  --steps 150000 \
  --eval-interval 2000 \
  --mask-illegal-policy \
  --stockfish-injection-path data/stockfish_injection \
  --stockfish-injection-games 300 \
  --stockfish-injection-interval 256 \
  --self-play-warmup-steps 15000 \
  --warmstart-buffer-size 300 \
  --warmstart-sample-frac 0.4 \
  --material-value-weight 0.5 \
  --material-value-anneal-frac 0.6 \
  --use-material-head \
  --material-head-loss-weight 0.25 \
  --root-terminal-draws \
  --root-terminal-draws-min-repeats 2 \
  --resign-enabled \
  --num-simulations 800 \
  --num-self-play-games 512 \
  --num-parallel-games 128 \
  --selfplay-q-ratio 0.25 \
  --resume checkpoints/chess/2026_06_23_warmstart_material/checkpoint_15000.pt \
  2>&1 | tee logs/2026_06_24_warmstart_qboot_s800.log
