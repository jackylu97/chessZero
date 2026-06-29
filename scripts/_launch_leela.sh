#!/bin/bash
# LEELA-FAITHFUL relabel run — the first FAIR test of the relabel approach with the
# missing-5-piece-tables bug fixed (prior runs relabeled only 2% of 5-piece plies).
#   - ON-POLICY: no search-side policy steering (tb_steer_policy=False default). The
#     model plays its own endgames; the TB signal enters ONLY via relabels of
#     self-play-reached states.
#   - VALUE relabel = pure Syzygy WDL (tb_value_dtz_shape=0.0); distance now comes
#     from the moves-left head, not from shaping the value (Leela design).
#   - MOVES-LEFT relabel = Gaviota DTM (tb_moves_left_weight=1.0) — the distance
#     gradient the WDL value head provably can't supply; head already in MCTS.
#   - selfplay_q_ratio=0 so relabels dominate at TB plies.
# Fresh from the 15k warmstart anchor (clean A/B vs tb_probe/tb_value/s09). No endgame
# SEEDING yet — that's the next build (Phase 1). This isolates the relabel approach.
cd /workspace/chessZero

RESUME=checkpoints/chess/2026_06_23_warmstart_material/checkpoint_15000.pt
if [ ! -f "$RESUME" ]; then
  echo "ERROR: warmstart anchor $RESUME not found"; exit 1
fi
echo "Starting leela_relabel FRESH from the warmstart anchor: $RESUME"

.venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_small \
  --run-id 2026_06_26_leela_relabel \
  --device cuda \
  --steps 150000 \
  --eval-interval 2000 \
  --mask-illegal-policy \
  --stockfish-injection-path data/stockfish_injection \
  --stockfish-injection-games 300 \
  --stockfish-injection-interval 256 \
  --self-play-warmup-steps 15000 \
  --warmstart-buffer-size 300 \
  --warmstart-sample-frac 0.4 \
  --warmstart-sample-frac-final 0.1 \
  --warmstart-anneal-frac 0.6 \
  --material-value-weight 0.5 \
  --material-value-anneal-frac 0.6 \
  --use-material-head \
  --material-head-loss-weight 0.25 \
  --root-terminal-draws \
  --root-terminal-draws-min-repeats 2 \
  --resign-enabled \
  --resign-holdout-frac 0.20 \
  --num-simulations 400 \
  --num-self-play-games 512 \
  --num-parallel-games 512 \
  --reanalyze-interval 0 \
  --tb-root-probe \
  --tb-path data/syzygy \
  --tb-gaviota-path data/gaviota \
  --tb-value-weight 1.0 \
  --tb-value-dtz-shape 0.0 \
  --tb-moves-left-weight 1.0 \
  --resume "$RESUME" \
  2>&1 | tee logs/2026_06_26_leela_relabel.log
