#!/bin/bash
# First full self-play run on the validated ATTENTION backbone, trained FROM SCRATCH on the
# Stockfish warmstart (15k warmup) + the decisive-signal stack. Modeled on _launch_egonly.sh
# but fresh (no --resume; old anchors are conv-arch / incompatible) and seeding from the
# endgame TRAIN split (15% of endgame FENs held out in data/endgame_seeds_holdout.txt for a
# clean generalization eval). Attention is baked into the chess_small preset now.
set -uo pipefail
cd /workspace/chessZero
RUN=2026_06_30_attn_warmstart_fix
tmux kill-session -t selfplay 2>/dev/null
tmux new-session -d -s selfplay "PYTHONPATH=. .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_small --run-id $RUN --device cuda \
  --steps 150000 --eval-interval 2000 --mask-illegal-policy \
  --stockfish-injection-path data/stockfish_injection \
  --stockfish-injection-games 300 --stockfish-injection-interval 256 \
  --self-play-warmup-steps 15000 --warmstart-buffer-size 300 \
  --warmstart-sample-frac 0.4 --warmstart-sample-frac-final 0.1 --warmstart-anneal-frac 0.6 \
  --material-value-weight 0.5 --material-value-anneal-frac 0.6 \
  --use-material-head --material-head-loss-weight 0.25 \
  --root-terminal-draws --root-terminal-draws-min-repeats 2 \
  --resign-enabled --resign-holdout-frac 0.20 \
  --num-simulations 400 --num-self-play-games 512 --num-parallel-games 512 \
  --reanalyze-interval 0 \
  --tb-root-probe --tb-path data/syzygy --tb-gaviota-path data/gaviota \
  --tb-relabel-workers 8 \
  --tb-value-weight 1.0 --tb-value-dtz-shape 0.5 --tb-moves-left-weight 1.0 \
  --ml-slope 0.02 --ml-max-effect 0.3 \
  --endgame-seed-frac 0.5 --endgame-seed-archive data/endgame_seeds_train.txt \
  2>&1 | tee /workspace/chessZero/selfplay_attn_warmstart_fix.log"
echo "launched tmux 'selfplay' (run $RUN); attach: tmux attach -t selfplay"
