#!/bin/bash
# Scaled winning architecture (EXP C: rep+dyn attention, smolgen, ssl+inv; L6, 200k seqs)
# + PRED-ATTENTION (shared 2-layer smolgen body before the policy/value split, re-attending
# at every MCTS node). Tests whether the prediction head has room before scaling further.
# Apples-to-apples baseline = the scaled run (0.41 term-ON conversion @ 36k).
# Log to the persistent volume so the curve survives a pod expiry this time.
cd /workspace/chessZero
USE_ATTENTION=1 \
USE_SMOLGEN=1 \
USE_DYN_ATTENTION=1 \
USE_PRED_ATTENTION=1 \
USE_CONSISTENCY=1 \
USE_INVERSE=1 \
ATTN_LAYERS=6 \
DATA=data/tb5_seq_big.pkl \
STEPS=40000 \
TAG_SUFFIX=_scaledL6 \
PYTHONPATH=. nohup .venv/bin/python scripts/train_tb_endgame.py \
  > /workspace/chessZero/scaled_predattn.log 2>&1 &
echo "PID $!"
