#!/bin/bash
# 2026_07_06_hybrid_minimal: the THEORY-CLEAN scale-compensator ablation
# (task #18; user hypothesis: self-play works at scale — which compensators
# are load-bearing at OUR scale?).
# KEEP:  Gumbel (the improvement guarantee) + uniform sampling + FILL + TB
#        value/DTM relabel (truth restoration for the measured draw-basin,
#        2026-06-17) + hard-z + 8-plane reward head + hybrid arch + warmstart.
# DROP:  seeds (+ exemption + curriculum), anchor, DAgger policy relabel,
#        merged batch (no seeds to merge), symmetry (already out).
# KEEP opening ε-mixture (self-play diversity, theory-neutral plumbing).
# Fresh start, same structure as v3 (15k SF warmstart + selfplay).
# Success compare @30k vs v3 AND control (0.152): conversion (Gumbel@400,
# REWARD_PLANES=8), reward precision, eval band, SF-agreement.
# minimal ~= v3  => production goes minimal (scaffolding not load-bearing).
# minimal <<     => scaffolding earned its place; production = v3 config.
set -uo pipefail
cd /workspace/chessZero
RUN=2026_07_06_hybrid_minimal
tmux kill-session -t selfplay 2>/dev/null
tmux new-session -d -s selfplay "PYTHONPATH=. .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_hybrid --run-id $RUN --device cuda \
  --reward-head-planes 8 \
  --use-gumbel --gumbel-m 16 --per-alpha 0 \
  --opening-mix-mean-plies 6 \
  --tb-value-hard \
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
  --tb-rollout-fill \
  2>&1 | tee /workspace/chessZero/selfplay_hybrid_minimal.log"
echo "launched tmux 'selfplay' (run $RUN, MINIMAL ARM); attach: tmux attach -t selfplay"
