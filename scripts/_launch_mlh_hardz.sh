#!/usr/bin/env bash
# 2026_06_28_mlh_hardz: Lc0-style HARD WDL value at TB plies. From the clean 15k warmstart
# anchor (same A/B start as mlh_from15k). Changes vs mlh_from15k:
#   --tb-value-hard            : one-hot win/draw/loss at TB plies (was soft eval_to_wdl,
#                                capped W-L ~0.88) → saturates Q near ±1, crisp win/draw
#                                separation (refutes throws), activates the |Q|-gated MLH.
#   --tb-value-dtz-shape 0.0   : drop the in-value distance gradient (distance now via MLH).
#   --ml-threshold 0.3         : matched to our value scale (was 0.8, Lc0's saturated-value
#                                threshold; now the hard value saturates so the MLH fires).
# Warmstart stays SOFT (eval_to_wdl on Stockfish per-position evals — genuinely uncertain).
# Deferred pooled relabel (8 workers).
set -euo pipefail
cd /workspace/chessZero
RUN=2026_06_28_mlh_hardz
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
  --num-simulations 400 --num-self-play-games 512 --num-parallel-games 512 \
  --reanalyze-interval 0 \
  --tb-root-probe --tb-path data/syzygy --tb-gaviota-path data/gaviota \
  --tb-relabel-workers 8 \
  --tb-value-weight 1.0 --tb-value-hard --tb-value-dtz-shape 0.0 --tb-moves-left-weight 1.0 \
  --tb-policy-weight 1.0 --tb-policy-weight-final 0.0 --tb-policy-anneal-frac 0.0 \
  --tb-dtz-weight 0.5 --tb-policy-temp 0.10 \
  --moves-left-head-planes 8 --moves-left-head-blocks 1 \
  --ml-threshold 0.3 --ml-slope 0.0027 --ml-max-effect 0.0345 \
  --endgame-seed-frac 1.0 --endgame-seed-archive data/endgame_seeds.txt \
  --warmstart-body "$ANCHOR"
