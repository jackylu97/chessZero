#!/bin/bash
# Winning EXP C architecture (rep+dyn attention, smolgen, ssl+inv; L6, 200k seqs) with the
# MOVES-LEFT HEAD FIX: linear DTM encoding (no value-style sqrt scalar_transform) + support 24,
# restoring 1-ply resolution across the long-mate range the sqrt+size-10 squash destroyed.
# Baseline to beat = the scaled run (0.41 term-ON conversion @ 36k, KRvK/KPvK ~0.17).
# Runs in tmux 'mllin24' with full live printout + tee to the persistent volume.
SESSION=mllin24
tmux kill-session -t "$SESSION" 2>/dev/null
tmux new-session -d -s "$SESSION" "cd /workspace/chessZero && \
USE_ATTENTION=1 USE_SMOLGEN=1 USE_DYN_ATTENTION=1 \
USE_CONSISTENCY=1 USE_INVERSE=1 \
ATTN_LAYERS=6 DATA=data/tb5_seq_big.pkl STEPS=40000 \
ML_LINEAR=1 ML_SUPPORT=24 \
TAG_SUFFIX=_scaledL6 \
PYTHONPATH=. .venv/bin/python -u scripts/train_tb_endgame.py 2>&1 | tee /workspace/chessZero/scaled_mllin24.log"
echo "launched tmux session '$SESSION' (attach: tmux attach -t $SESSION)"
