#!/usr/bin/env bash
# Endgame-only MECHANICS TEST (2026_06_27_egonly): seed_frac=1.0 (every self-play game is a
# tablebase endgame) to isolate the conversion mechanic from the midgame draw basin. Turns ON
# the two progress signals we diagnosed as missing: dtz_shape=0.5 (distance gradient in the
# VALUE) + a stronger moves-left MCTS utility (ml_slope 0.02 / ml_max_effect 0.3, ~4x/3x the
# default). ml_threshold left at the 0.3 default. Resumes the 15k warmstart anchor. Validation
# targets: seed/mate_rate climbing past ~25-30%, value |Q| on won positions saturating (gate
# fires), self_play draw_rate / ply-cap falling. NOT a deployable model (over-specializes to
# endgames) — a mechanics probe; the 'real' run tones seed_frac back to ~0.3.
set -euo pipefail
cd /workspace/chessZero
RUN=2026_06_27_egonly
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
  --tb-value-weight 1.0 --tb-value-dtz-shape 0.5 --tb-moves-left-weight 1.0 \
  --ml-slope 0.02 --ml-max-effect 0.3 \
  --endgame-seed-frac 1.0 --endgame-seed-archive data/endgame_seeds.txt \
  --resume checkpoints/chess/2026_06_23_warmstart_material/checkpoint_15000.pt
