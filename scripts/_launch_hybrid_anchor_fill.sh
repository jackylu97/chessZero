#!/bin/bash
# 2026_07_03_hybrid_anchor_fill: the ~2x HYBRID architecture (chess_hybrid preset:
# conv-SE stem + d128 attention body in rep+dyn, shared 1-layer attention
# prediction body, widened value/ML heads — 5.57M inference params vs 2.55M) on
# the SAME signal stack as 2026_07_03_conv_anchor_fill. Single-variable arch A/B:
#   conv_anchor_fill  (conv,   resumed conv 15k warmstart)   vs
#   hybrid_anchor_fill(hybrid, fresh 15k warmstart phase — no hybrid ckpt exists;
#                      the conv run's resume point was itself a 15k warmstart on
#                      the same Stockfish pool, so lineages are equivalent)
# Signal stack (matched to the conv run exactly): anchor 64/256, rollout fill,
# soft TB value dtz-shape 0.0, DTM relabel 1.0, seed 0.30 (train split),
# resign + 0.20 holdout, material shaping, 400 sims. ML-utility at preset
# defaults. Watch: eval/win_rate_vs_random vs BOTH conv_anchor_fill and seed30
# at matched steps; raw diag at 30k (CFG=chess_hybrid USE_ATTENTION=1
# USE_DYN_ATTENTION=1 USE_PRED_ATTENTION=1).
set -uo pipefail
cd /workspace/chessZero
RUN=2026_07_03_hybrid_anchor_fill
tmux kill-session -t selfplay 2>/dev/null
tmux new-session -d -s selfplay "PYTHONPATH=. .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_hybrid --run-id $RUN --device cuda \
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
  --tb-value-weight 1.0 --tb-value-dtz-shape 0.0 --tb-moves-left-weight 1.0 \
  --endgame-seed-frac 0.30 --endgame-seed-archive data/endgame_seeds_train.txt \
  --tb-anchor-path data/tb_anchor --tb-anchor-games 64 --tb-anchor-interval 256 \
  --tb-rollout-fill \
  2>&1 | tee /workspace/chessZero/selfplay_hybrid_anchor_fill.log"
echo "launched tmux 'selfplay' (run $RUN); attach: tmux attach -t selfplay"
