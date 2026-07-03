#!/bin/bash
# 2026_07_04_hybrid_gumbel: hybrid v2 stack + the two precedent-audit levers
# (setup_vs_precedent_2026_07_03.md R1 + R9), per user direction — combined,
# not ablated, to prove the config for the big-GPU production run:
#   R1  --per-alpha 0: UNIFORM buffer sampling (MuZero-board/KataGo/Lc0
#       3-way consensus; prevents value-TD priorities from starving anchor
#       games whose easy value targets hide unlearned policy content).
#   R9  --use-gumbel: Plain Gumbel MuZero at the root in TensorMCTS
#       (Gumbel Top-m candidates, Sequential Halving, completed-Q π' targets,
#       argmax g+logits+σ(q) selection). Replaces Dirichlet noise + the
#       temperature schedule at the root; guaranteed policy improvement per
#       simulation given value signal — which anchor+fill now supply (the
#       draw-basin-era gumbel failure was the precondition, not the method).
#       Oracle-parity-tested vs the numpy implementation (test_gumbel_tensor_mcts).
# Resumes the same v1 15k warmstart checkpoint as v2 (matched lineage);
# seed-resign exemption kept. Sims stay 400 (comparability first; the low-sim
# dividend is banked for the big-GPU run).
set -uo pipefail
cd /workspace/chessZero
RUN=2026_07_04_hybrid_gumbel
tmux kill-session -t selfplay 2>/dev/null
tmux new-session -d -s selfplay "PYTHONPATH=. .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_hybrid --run-id $RUN --device cuda \
  --resume checkpoints/chess/2026_07_03_hybrid_anchor_fill/checkpoint_15000.pt \
  --use-gumbel --gumbel-m 16 --per-alpha 0 \
  --resign-exempt-seeded \
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
  2>&1 | tee /workspace/chessZero/selfplay_hybrid_gumbel.log"
echo "launched tmux 'selfplay' (run $RUN); attach: tmux attach -t selfplay"
