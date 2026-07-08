#!/bin/bash
# First full self-play run with the validated attention backbone + the decisive-signal
# levers. chess_small cold start (attention baked into the preset now). Endgame engine =
# tb_root_probe (ground-truth DTZ steering) + endgame seeding. tmux + tee to persistent log.
# TB native TensorBoard lands in runs/chess/<run-id> (picked up by the existing `tb` session).
SESSION=selfplay
RUNID=2026_06_30_attn_decisive
tmux kill-session -t "$SESSION" 2>/dev/null
tmux new-session -d -s "$SESSION" "cd /workspace/chessZero && \
PYTHONPATH=. .venv/bin/python -u scripts/train.py \
  --game chess_small \
  --run-id $RUNID \
  --device cuda \
  --steps 150000 \
  --tb-root-probe \
  --endgame-seed-frac 0.25 \
  --endgame-seed-archive data/endgame_seeds.txt \
  --resign-enabled \
  2>&1 | tee /workspace/chessZero/selfplay_attn_decisive.log"
echo "launched tmux '$SESSION' (run-id $RUNID); attach: tmux attach -t $SESSION"
