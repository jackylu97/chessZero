#!/bin/bash
# =============================================================================
# PRODUCTION RUN — 2026-07-08 (see production_config_2026_07_08.md for
# rationale, production_context_2026_07_08.md for full campaign context)
#
# Week-long run. Every setting below is either measured-in-arm or a
# pre-registered conservative default; nothing is an inherited accident.
# =============================================================================
# SIMS: 800 (user decision 2026-07-08, endorsed): the compounding rule favors
#   conservative-high search; the only production-scale datapoint (the 90%
#   run) used 800; static 800-vs-400 flatness is not valid pricing evidence.
#   If wall-clock forces a choice, KEEP 800 and reduce games/round.
# GAMES/ROUND: 2026-07-09 — cut 1024->512 (resumed @ checkpoint_31000). XL
#   dynamics-attention self-play is ~10x chess_small: at 1024 games/800 sims one
#   round is ~2h and BLOCKS training -> ~94-day ETA. Halving -> ~7-8wk, reuse
#   ~1.65->3.3. Kept 800 sims + max_plies per user (further trims cost quality).
#   See memory prod-run-halved-games-selfplay. Resume via RESUME=<ckpt.pt> knob.
# HARDWARE KNOBS (set for the production GPU before launching):
#   PAR_GAMES : parallel self-play games. 512 fits 32GB alongside training;
#               1024 recommended on >=48GB (true replay-ratio reduction).
#   N_GAMES   : games per round. Keep == PAR_GAMES (one chunk) or 2x for
#               two sequential chunks on smaller cards.
#   BUFFER    : replay buffer (games). Scale with games/round to hold
#               passes-per-position near ~3 (5120 @512, 10240 @1024).
#   BATCH is fixed 512 via config; grad-checkpointing keeps XL training
#   inside 32GB — remove --grad-checkpoint-attention on >=48GB for ~25%
#   faster steps.
PAR_GAMES="${PAR_GAMES:-512}"
N_GAMES="${N_GAMES:-512}"
BUFFER="${BUFFER:-5120}"
STEPS="${STEPS:-600000}"
RUN="${RUN:-2026_07_08_production}"
RESUME="${RESUME:-}"   # optional: path to checkpoint_*.pt to resume from (buffer auto-loaded from sibling .buf)
set -uo pipefail
cd "$(dirname "$0")/.."
tmux kill-session -t production 2>/dev/null
tmux new-session -d -s production "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_hybrid_xl --run-id $RUN --device cuda \
  ${RESUME:+--resume $RESUME} \
  --policy-head-type from_to \
  --reward-head-planes 8 \
  --use-gumbel --gumbel-m 16 --per-alpha 0 \
  --num-simulations 800 \
  --num-self-play-games $N_GAMES --num-parallel-games $PAR_GAMES \
  --replay-buffer-size $BUFFER \
  --steps $STEPS --eval-interval 2000 --mask-illegal-policy \
  --self-play-warmup-steps 30000 --warmstart-buffer-size 300 \
  --stockfish-injection-path data/stockfish_injection \
  --stockfish-injection-games 300 --stockfish-injection-interval 256 \
  --warmstart-sample-frac 0.4 --warmstart-sample-frac-final 0.1 --warmstart-anneal-frac 0.6 \
  --batch-mixture-schedule '[[0.00,{\"warmstart\":0.70,\"anchor\":0.30,\"selfplay\":0.00}],[0.10,{\"warmstart\":0.40,\"anchor\":0.15,\"selfplay\":0.45}],[0.40,{\"warmstart\":0.20,\"anchor\":0.10,\"selfplay\":0.70}],[0.60,{\"warmstart\":0.10,\"anchor\":0.10,\"selfplay\":0.80}]]' \
  --anchor-max-size 1024 \
  --merged-seed-batch \
  --opening-mix-mean-plies 6 \
  --seed-curriculum \
  --tb-value-hard \
  --tb-policy-weight 0.5 --tb-policy-weight-final 0.2 --tb-policy-anneal-frac 0.6 \
  --material-value-weight 0.5 --material-value-anneal-frac 0.6 \
  --use-material-head --material-head-loss-weight 0.25 \
  --root-terminal-draws --root-terminal-draws-min-repeats 2 \
  --resign-enabled --resign-holdout-frac 0.20 --resign-exempt-seeded \
  --reanalyze-interval 0 \
  --tb-root-probe --tb-path data/syzygy --tb-gaviota-path data/gaviota \
  --tb-relabel-workers 8 \
  --tb-value-weight 1.0 --tb-value-dtz-shape 0.0 --tb-moves-left-weight 1.0 \
  --endgame-seed-frac 0.30 --endgame-seed-archive data/endgame_seeds_train.txt \
  --tb-anchor-path data/tb_anchor --tb-anchor-games 64 --tb-anchor-interval 256 \
  --tb-rollout-fill \
  2>&1 | tee selfplay_production.log"
echo "PRODUCTION launched in tmux 'production' (run $RUN)"
echo "Attach:  tmux attach -t production"
echo "Probes:  bash scripts/prod_probes.sh $RUN   (run in a second tmux; every 10k steps)"
