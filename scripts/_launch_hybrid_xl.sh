#!/bin/bash
# 2026_07_06_hybrid_xl: the 5x SCALE TEST (strategy §16). v3 config EXACTLY,
# two deltas only: chess_hybrid_xl (23.97M inference params, 4.9x — d288,
# 6-layer bodies, 8 heads, 3-layer pred body, 3-block stems, fc256) and the
# 2x schedule (30k warmstart / 300k total; anneal FRACTIONS scale with
# training_steps automatically). Buffer deliberately unchanged (5120) for a
# strict scaling read — SF-agreement is the big-model-small-buffer overfit
# tripwire. Verdict point: 60k (30k warm + 30k selfplay, structurally matched
# to the three arms' 30k), ~2.5-3 days on the 5090.
# Registered predictions (board-game scaling laws, Jones 2021):
#   1. milestones left-shifted in steps (sample efficiency of the larger net)
#   2. 60k verdict should exceed v3 (evals ~59, conversion 0.113/0.160-long)
#   3. still-climbing at 60k => extend, do not conclude
# 2026-07-06 MID-RUN CHANGE at 32k (user decision): sims 400 -> 200 (Jones
# train/test-compute trade; 800-vs-400 diag showed depth is not the binder;
# Gumbel guarantee holds at low sims). Verdict diags stay SIMS=400 for
# cross-arm comparability. XL@200 is the production-rehearsal config.
# 2026-07-07 SECOND CHANGE at ~62k: sims BACK to 400 (A/B/A). Evidence: under
# 200-sim training, general strength soared (eval record, SF-agreement 0.814)
# but 400-sim-measured conversion REGRESSED 0.107 -> 0.047 and the sims sweep
# equalized at the LOW level (net stopped converting regardless of depth).
# Reading: self-play search depth is load-bearing for the TECHNIQUE-learning
# loop while the net is young — search finds conversions, the net banks them;
# at 200 sims the loop starves. Watch conversion recover by ~75k to confirm.
set -uo pipefail
cd /workspace/chessZero
RUN=2026_07_06_hybrid_xl
tmux kill-session -t selfplay 2>/dev/null
tmux new-session -d -s selfplay "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_hybrid_xl --run-id $RUN --device cuda \
  --resume checkpoints/chess/2026_07_06_hybrid_xl/checkpoint_62000.pt \
  --grad-checkpoint-attention \
  --reward-head-planes 8 \
  --use-gumbel --gumbel-m 16 --per-alpha 0 \
  --resign-exempt-seeded \
  --merged-seed-batch \
  --opening-mix-mean-plies 6 \
  --seed-curriculum \
  --tb-value-hard \
  --tb-policy-weight 0.5 --tb-policy-weight-final 0.2 --tb-policy-anneal-frac 0.6 \
  --steps 300000 --eval-interval 2000 --mask-illegal-policy \
  --stockfish-injection-path data/stockfish_injection \
  --stockfish-injection-games 300 --stockfish-injection-interval 256 \
  --self-play-warmup-steps 30000 --warmstart-buffer-size 300 \
  --warmstart-sample-frac 0.4 --warmstart-sample-frac-final 0.1 --warmstart-anneal-frac 0.6 \
  --material-value-weight 0.5 --material-value-anneal-frac 0.6 \
  --use-material-head --material-head-loss-weight 0.25 \
  --root-terminal-draws --root-terminal-draws-min-repeats 2 \
  --resign-enabled --resign-holdout-frac 0.20 \
  --num-simulations 400 --num-self-play-games 512 --num-parallel-games 512 \
  --reanalyze-interval 0 \
  --tb-root-probe --tb-path data/syzygy --tb-gaviota-path data/gaviota \
  --tb-relabel-workers 8 \
  --tb-value-weight 1.0 --tb-value-dtz-shape 0.0 --tb-moves-left-weight 1.0 \
  --endgame-seed-frac 0.30 --endgame-seed-archive data/endgame_seeds_train.txt \
  --tb-anchor-path data/tb_anchor --tb-anchor-games 64 --tb-anchor-interval 256 \
  --tb-rollout-fill \
  2>&1 | tee /workspace/chessZero/selfplay_hybrid_xl.log"
echo "launched tmux 'selfplay' (run $RUN, 5x SCALE TEST, fresh start); attach: tmux attach -t selfplay"
