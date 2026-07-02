#!/bin/bash
# 2026_07_02_anchor_fill: first run with the TWO new decisive-signal channels
# (strategy_2026_07_02.md) on the attention backbone:
#   1. TB ANCHOR (--tb-anchor-*): tablebase-optimal demonstration games
#      (data/tb_anchor, gen_tb_anchor_games.py) injected into the rolling buffer
#      64 games / 256 steps, cycling forever — the validated supervised-proxy
#      signal (KQvK 0.91) as a persistent TB→policy teaching channel.
#   2. TB ROLLOUT FILL (--tb-rollout-fill): won-but-unconverted NON-seeded games
#      truncated at their first decisive in-TB ply + finished with TB-optimal
#      play — true decisive z for the WHOLE trajectory (the win adjudication the
#      per-ply value relabel can't propagate to pre-TB plies) + on-distribution
#      conversion demonstrations. Replaces what search steering actually provided
#      (sustained decisive buffer) without off-policy search bias.
# Changes vs _launch_attn_warmstart.sh (the 06-30 baseline):
#   - resume from its checkpoint_16000 (post-warmstart, pre-erosion) + its .buf;
#     --reset-injection-cursor re-fills the warm anchor pool from shard 0.
#   - --tb-value-hard --tb-value-dtz-shape 0.0 (proxy-faithful hard WDL one-hot
#     at TB plies; hard mode thresholds on sign so the shape MUST be 0 — a
#     dtz-shaped deep win at 0.5 would mislabel as draw).
#   - --endgame-seed-frac 0.3 (was 0.5): restore midgame data share; the anchor
#     now carries the dense endgame signal, seeds are on-policy practice + the
#     seed/* eval metrics.
# Leela formulation unchanged: on-policy search (no steering), Syzygy value
# relabel + Gaviota DTM moves-left relabel, moves-left MCTS utility.
# Watch: eval/win_rate_vs_random (crutch-free), self_play/tb_fill_rate,
# seed/mate_rate, buffer_decisive_frac, resign_false_positive_rate (should fall
# well below 1.0 as real conversion skill arrives), raw diag on checkpoints.
set -uo pipefail
cd /workspace/chessZero
RUN=2026_07_02_anchor_fill
tmux kill-session -t selfplay 2>/dev/null
tmux new-session -d -s selfplay "PYTHONPATH=. .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_small --run-id $RUN --device cuda \
  --resume checkpoints/chess/2026_06_30_attn_warmstart_fix/checkpoint_16000.pt \
  --reset-injection-cursor \
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
  --tb-value-weight 1.0 --tb-value-hard --tb-value-dtz-shape 0.0 --tb-moves-left-weight 1.0 \
  --ml-slope 0.02 --ml-max-effect 0.3 \
  --endgame-seed-frac 0.3 --endgame-seed-archive data/endgame_seeds_train.txt \
  --tb-anchor-path data/tb_anchor --tb-anchor-games 64 --tb-anchor-interval 256 \
  --tb-rollout-fill \
  2>&1 | tee /workspace/chessZero/selfplay_anchor_fill.log"
echo "launched tmux 'selfplay' (run $RUN); attach: tmux attach -t selfplay"
