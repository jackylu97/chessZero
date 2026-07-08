#!/usr/bin/env bash
# 2026_06_27_mlh_from15k: the MLH-progress recipe started from the CLEAN 15k warmstart
# anchor (NOT resumed from a draw-basin checkpoint) so ALL endgame self-play happens under
# the new recipe — directly comparable to leela/seed30/polrelabel which plateaued at ~5-10%
# conversion from this exact anchor. Same warmstart handling as those (empty buffer +
# stockfish injection). Fixes vs the old recipe: (1) MLH search utility = Lc0 MEvaluator
# (progress delta, |Q|>0.8 gate, Q-scaled, cap 0.0345); (2) widen the moves-left head input
# 1->8 planes +1 block so it can LEARN DTM; (3) steeper DTZ-optimal policy target; value head
# kept WDL. Body warm-start from the anchor (body/policy/value transfer, widened MLH reinit;
# fresh optimizer). Deferred pooled TB relabel (8 workers).
set -euo pipefail
cd /workspace/chessZero
RUN=2026_06_27_mlh_from15k
ANCHOR=checkpoints/chess/2026_06_23_warmstart_material/checkpoint_15000.pt
[ -f "$ANCHOR" ] || { echo "ERROR: anchor $ANCHOR not found"; exit 1; }
exec .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_small --run-id "$RUN" --device cuda \
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
  --tb-policy-weight 1.0 --tb-policy-weight-final 0.0 --tb-policy-anneal-frac 0.0 \
  --tb-dtz-weight 0.5 --tb-policy-temp 0.10 \
  --moves-left-head-planes 8 --moves-left-head-blocks 1 \
  --ml-threshold 0.8 --ml-slope 0.0027 --ml-max-effect 0.0345 \
  --endgame-seed-frac 1.0 --endgame-seed-archive data/endgame_seeds.txt \
  --warmstart-body "$ANCHOR"
