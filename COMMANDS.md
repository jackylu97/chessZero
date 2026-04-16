# ChessZero — Quick Reference

## Training

```bash
# Train tictactoe with default config (5000 steps, auto-detects GPU)
python scripts/train.py --game tictactoe

# Override steps
python scripts/train.py --game tictactoe --steps 2000

# Resume from a checkpoint
python scripts/train.py --game tictactoe --resume checkpoints/tictactoe/checkpoint_1000.pt

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

```bash
python scripts/evaluate.py --game tictactoe --checkpoint checkpoints/tictactoe/checkpoint_5000.pt --mode play
```

You play as O (player -1). Enter moves as `row col` (e.g. `2 3` for row 2, column 3).
