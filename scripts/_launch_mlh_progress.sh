#!/usr/bin/env bash
# 2026_06_27_mlh_progress: ENDGAME PROGRESS recipe (Lc0/KataGo-validated). Diagnosis:
# net preserves the win but can't make PROGRESS — policy target ~uniform over win moves
# (entropy 2.21=floor), WDL value flat at +1, MLH can't supply distance (head→DTZ 0.035 vs
# latent-MLP 0.80) → shuffles (27%) / liquidates to draw (73%). Fixes:
#   (1) MLH search utility = Lc0 MEvaluator: (child_m-parent_m) PROGRESS, |Q|>0.8 gate,
#       Q-scaled, small cap 0.0345 (was: absolute child_m, |Q|>0.3, no scale, cap 0.1).
#   (2) widen the moves-left head input 1->8 planes +1 residual block so it can LEARN DTM.
#   (3) steepen the soft policy target toward the DTZ-optimal (progressing) move
#       (dtz_weight 0.05->0.5, temp 0.15->0.10) — Stockfish: DTZ is the rule-aware metric.
#   value head kept WDL (Lc0+KataGo+our probe: distance belongs in the distance head).
# Body warm-start from w10sharp ckpt_40000 (body/policy/value transfer, MLH head reinit).
# Deferred + pooled TB relabel (8 workers) off the self-play hot path (~3.6x).
set -euo pipefail
cd /workspace/chessZero
RUN=2026_06_27_mlh_progress
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
  --warmstart-body checkpoints/chess/2026_06_27_s100_w10sharp/checkpoint_40000.pt
