#!/bin/bash
# TB-VALUE arm: identical to _launch_tbprobe.sh (400 sims / 512 games / 256 parallel)
# but ADDS DTZ value-target relabeling (tb_value_weight=1.0, tb_value_dtz_shape=0.5)
# on top of the search-side probe. Resumes from the LATEST tb_probe baseline
# checkpoint so it's a same-model before/after test: does relabeling flip the value
# head (corr(value,-DTZ): -0.34 -> +) and unlock no-probe conversion? selfplay_q_ratio
# stays 0 (config default) so the TB value isn't washed out.
cd /workspace/chessZero

RESUME=$(ls -t checkpoints/chess/2026_06_25_tb_probe/checkpoint_*.pt 2>/dev/null | head -1)
if [ -z "$RESUME" ]; then
  echo "ERROR: no tb_probe baseline checkpoint to resume from"; exit 1
fi
echo "Resuming tb_value from baseline: $RESUME"

.venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_small \
  --run-id 2026_06_25_tb_value \
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
  --warmstart-sample-frac-final 0.1 \
  --warmstart-anneal-frac 0.6 \
  --material-value-weight 0.5 \
  --material-value-anneal-frac 0.6 \
  --use-material-head \
  --material-head-loss-weight 0.25 \
  --root-terminal-draws \
  --root-terminal-draws-min-repeats 2 \
  --resign-enabled \
  --resign-holdout-frac 0.20 \
  --num-simulations 400 \
  --num-self-play-games 512 \
  --num-parallel-games 256 \
  --reanalyze-interval 0 \
  --tb-root-probe \
  --tb-dtz-weight 1.0 \
  --tb-value-weight 1.0 \
  --tb-value-dtz-shape 0.5 \
  --resume "$RESUME" \
  2>&1 | tee logs/2026_06_25_tb_value.log
