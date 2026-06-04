#!/usr/bin/env bash
# One-shot setup for a RunPod (or any native-Linux) GPU box using a PyTorch
# BASE IMAGE — i.e. torch + CUDA + driver are already installed and matched.
# We deliberately do NOT reinstall torch (avoids a multi-GB download and any
# driver mismatch); we only add the small runtime deps on top of the image's
# Python, then Stockfish for the warmstart/eval scripts.
#
# Usage (from the repo root):
#   bash scripts/runpod_setup.sh
#
# Pick the NEWEST available runpod/pytorch template — this code is developed
# against torch >= 2.11. If the image ships older torch, setup still proceeds
# (with a warning); if something breaks, switch to a uv-pinned torch instead.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root, regardless of where this is invoked from
PY="$(command -v python3 || command -v python)"
echo "Using Python: $PY"

# 1. Sanity-check the image's torch BEFORE building on it — fail fast if the
#    template has no CUDA (wrong template chosen).
"$PY" - <<'PY'
import sys
try:
    import torch
except ModuleNotFoundError:
    sys.exit("ERROR: no torch in this image — pick a RunPod *PyTorch* template, "
             "or use the uv-pinned-torch setup instead.")
if not torch.cuda.is_available():
    sys.exit("ERROR: torch is present but CUDA is unavailable — the driver/template "
             "does not expose a GPU. Pick a GPU PyTorch template.")
v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
print(f"image torch {torch.__version__}  cuda {torch.version.cuda}  gpu {torch.cuda.get_device_name(0)}")
if v < (2, 11):
    print(f"WARNING: image torch {torch.__version__} < 2.11 — code is developed "
          "against 2.11; if you hit API errors, pin torch via uv instead.")
PY

# 2. Fast installer (uv). Use it if present, else grab it.
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  "$PY" -m pip install -q uv
fi

# 3. Runtime deps EXCEPT torch — installed into the image's Python.
#    Mirrors pyproject.toml [project.dependencies] minus torch.
echo "Installing runtime deps (excluding torch)..."
uv pip install --python "$PY" \
  "numpy>=2.2.6" \
  "psutil>=5.9.0" \
  "python-chess>=1.999" \
  "tensorboard>=2.20.0" \
  "tqdm>=4.67.3" \
  "flask>=3.0.0"

# Optional: test deps. Uncomment to run the test suite on the box.
# uv pip install --python "$PY" "pytest>=9.0.3"

# 4. Stockfish — a system binary (NOT a pip dep), only needed by the
#    warmstart-generation + eval/health scripts. apt lands it at
#    /usr/games/stockfish AND on PATH, satisfying both default lookups
#    (generate_stockfish_games.py uses "stockfish"; eval_checkpoint_health.py
#    uses "/usr/games/stockfish").
if ! command -v stockfish >/dev/null 2>&1 && [ ! -x /usr/games/stockfish ]; then
  echo "Installing stockfish..."
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq stockfish || \
      echo "WARNING: stockfish install failed — core training is fine; "\
           "warmstart/eval scripts will need it on PATH."
  else
    echo "WARNING: no apt-get — install stockfish manually for warmstart/eval."
  fi
fi

echo ""
echo "Setup complete. Verify and launch from the repo root:"
echo "  $PY -c 'import torch,chess,numpy,psutil,tensorboard,tqdm; print(\"imports OK\")'"
echo "  ./scripts/supervise_train.sh --game chess --run-id <run-id>"
echo ""
echo "NOTE: on native Linux DROP the WSL-only '--tensor-mcts-select-backend eager'"
echo "flag — the default triton/inductor backend is faster and stable off WSL."
echo "Start TensorBoard separately:  tensorboard --logdir runs/chess --port 6006 --bind_all"
