# Running ChessZero on RunPod (or any native-Linux GPU box)

Strategy: use a **RunPod PyTorch base image** (torch + CUDA + driver pre-matched) and
add only the small runtime deps on top. No multi-GB torch reinstall, no driver mismatch.

## Quickstart

1. **Launch a pod** with the *newest available* `runpod/pytorch` template and a GPU.
   - **GPU:** an **RTX 4090** is the cost/speed sweet spot for this workload (small 23M-param
     net, partially CPU-bound self-play); A100/H100 are overkill and far pricier per hour.
   - **Persistent volume:** mount one and put the repo on it so `checkpoints/` and `runs/`
     survive pod restarts.
   - Expose a port for TensorBoard (6006).

2. **Clone + set up** (from the repo root):
   ```bash
   git clone https://github.com/jackylu97/chessZero.git && cd chessZero
   bash scripts/runpod_setup.sh
   ```
   The script verifies the image's torch sees CUDA, installs the runtime deps (numpy,
   python-chess, tensorboard, tqdm, psutil, flask) into the image's Python, and apt-installs
   Stockfish (only needed for warmstart/eval).

3. **Verify:**
   ```bash
   python -c "import torch,chess,numpy,psutil,tensorboard,tqdm; print('imports OK', torch.cuda.is_available())"
   ```

4. **Launch training + TensorBoard** (from the repo root):
   ```bash
   ./scripts/supervise_train.sh --game chess --run-id <run-id>
   tensorboard --logdir runs/chess --port 6006 --bind_all
   ```

## Notes specific to leaving WSL

- **Drop `--tensor-mcts-select-backend eager`.** That flag was a workaround for a WSL-only
  autocast+autograd+inductor SIGSEGV. On native Linux the default **triton/inductor** backend
  is faster and stable — just omit the flag (the chess preset defaults to triton).
- The host BSODs you saw were a **failing CPU on the old box**, not the code — irrelevant here.

## If the image's torch is too old

This code targets **torch >= 2.11**. If the newest RunPod template still ships older torch and
you hit API errors, switch to the **uv-pinned-torch** path instead: add a `[tool.uv.index]`
for the CUDA 12.4 PyTorch wheel and `uv sync` the full environment (reproducible, but downloads
torch each cold start). Ask and this can be wired up.

## Dependency notes

- `pyproject.toml` is the source of truth; `runpod_setup.sh` installs its deps **minus torch**
  (torch comes from the image). Keep the two in sync if you add a dependency.
- `psutil` is a hard runtime dep (`trainer.py`) — it's now declared (was previously missing).
- Stockfish is a **system binary**, not a pip package; `uv`/`pip` will never install it.
