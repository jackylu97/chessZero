# ChessZero — Quick Reference

## Training

```bash
# Train tictactoe with default config (5000 steps, auto-detects GPU)
python scripts/train.py --game tictactoe

# Override steps
python scripts/train.py --game tictactoe --steps 2000

# Resume from a checkpoint
python scripts/train.py --game tictactoe --resume checkpoints/tictactoe/checkpoint_1000.pt

# Resume a chess warmstart run — re-pass --warmstart-path so the Stockfish games
# land back in the buffer (they're intentionally excluded from the .buf snapshot).
python scripts/train.py --game chess \
  --resume checkpoints/chess/<run-id>/checkpoint_<step>.pt \
  --warmstart-path data/stockfish_games

# Cap the .buf snapshot to the last N self-play games (in-memory buffer is unchanged).
# Use this when checkpoint_*.buf grows multi-GB and OOMs the host at save time.
python scripts/train.py --game chess --max-buf-save-games 2000

# Force a specific device
python scripts/train.py --game tictactoe --device cpu

# Custom TensorBoard log directory
python scripts/train.py --game tictactoe --log-dir my_runs
```

Checkpoints are saved to `checkpoints/<game>/checkpoint_<step>.pt`.
The replay buffer is saved alongside as `checkpoint_<step>.buf`.

## TensorBoard

```bash
tensorboard --logdir runs
# then open http://localhost:6006
```

Key metrics logged:
- `loss/total_loss`, `loss/policy_loss`, `loss/value_loss`, `loss/reward_loss`
- `eval/win_rate_vs_random`, `eval/draw_rate_vs_minimax` (tictactoe only)
- `self_play/avg_game_length`, `self_play/p1_win_rate`
- `reanalyze/positions_updated`
- `train/lr`, `train/per_beta`

## Evaluation

```bash
# Evaluate vs random opponent (100 games)
python scripts/evaluate.py --game tictactoe --checkpoint checkpoints/tictactoe/checkpoint_5000.pt

# Evaluate with more games
python scripts/evaluate.py --game tictactoe --checkpoint checkpoints/tictactoe/checkpoint_5000.pt --num-games 200
```

## Play Against the Model

### Terminal (tictactoe / connect4)

```bash
python scripts/evaluate.py --game tictactoe --checkpoint checkpoints/tictactoe/checkpoint_5000.pt --mode play
```

You play as O (player -1). Enter moves as `row col` (e.g. `2 3` for row 2, column 3).

### Web (chess)

```bash
python scripts/play_web.py --checkpoint checkpoints/chess/<run-id>/checkpoint_<step>.pt
# then open http://127.0.0.1:5000
```

You play Black; the model plays White. Drag pieces to move; pawn promotions default to queen. Click "New game" to reset.

Options: `--port 5001` (default 5000), `--host 0.0.0.0` (default 127.0.0.1), `--device cpu` (default auto).

## Browse Replay Buffer (Web)

Inspect the games stored in a saved `.buf` — useful for sanity-checking self-play (shuffle-draw rates, repetitions, capture counts).

```bash
python scripts/view_buffer_web.py --buffer checkpoints/chess/<run-id>/checkpoint_<step>.buf
# then open http://127.0.0.1:5000
```

Games are sorted by (captures asc, length asc) so the most pathological shuffle-draws surface first. Pick a game from the dropdown; use the slider or arrow keys to step through moves.

Options: `--port`, `--host`, `--rebuild-cache` (force re-decode the buffer's game records).

Note: both web commands default to port 5000. If you want to run them simultaneously, pass `--port` to one of them.
