#!/usr/bin/env bash
# 2026_06_27_polrelabel: the first run with the VALUE relabel actually wired (it was a
# silent no-op before — make_target never got tb_value_weight; commit fd2da19) PLUS the
# new soft TB POLICY relabel (Lc0 DTZ policy boost, safe for us sans KLDGain) to fix the
# policy prior that mass-loads the win-throwing move. Soft (win-preserving distribution),
# annealed to 0 over 60% of training so the teacher fades (anti probe-is-a-crutch). Back
# to seed_frac=0.3 (seed_frac=1.0 starved the policy and collapsed to draws). dtz_shape=0.5
# kept (value distance gradient). Default MLH (the move-decomp didn't support strengthening
# it). Resumes the 15k warmstart anchor. Watch: seed/mate_rate, value won-vs-DRAW margin.
set -euo pipefail
cd /workspace/chessZero
RUN=2026_06_27_polrelabel
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
  --tb-policy-weight 0.5 --tb-policy-weight-final 0.0 --tb-policy-anneal-frac 0.6 \
  --endgame-seed-frac 0.3 --endgame-seed-archive data/endgame_seeds.txt \
  --resume checkpoints/chess/2026_06_23_warmstart_material/checkpoint_15000.pt
