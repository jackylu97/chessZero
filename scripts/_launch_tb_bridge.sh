#!/bin/bash
cd /workspace/chessZero
exec env PYTHONPATH=. .venv/bin/python -u scripts/log_to_tb.py scaled_mllin24.log mllin24
