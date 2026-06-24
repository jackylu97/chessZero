"""Modal wrapper for ChessZero training — run the existing CLI on Modal GPUs.

Nothing in the training code changes; this just builds an image from uv.lock and
shells out to scripts/train.py with paths pointed at Modal Volumes.

────────────────────────────────────────────────────────────────────────────
ONE-TIME SETUP
────────────────────────────────────────────────────────────────────────────
  pip install modal        # in any env; the CLI is all you need
  modal token new          # auth (interactive — you must run this yourself)

  # Create the data volume and upload the warm-start assets ONCE (no regen):
  modal volume create chesszero-data
  modal volume put chesszero-data data/stockfish_injection  /stockfish_injection
  modal volume put chesszero-data data/endgame_seed         /endgame_seed
  modal volume put chesszero-data tools/stockfish           /stockfish
  modal volume put chesszero-data \
      checkpoints/chess/2026_06_23_warmstart_material/checkpoint_15000.pt \
      /checkpoint_15000.pt
  # (stockfish_injection is ~3.1 GB — one slow upload, then it's cached forever.)

────────────────────────────────────────────────────────────────────────────
RUN
────────────────────────────────────────────────────────────────────────────
  # 1. Validate the GPU stack first (torch+cu130, triton, chess_gpu) — no data:
  modal run modal_app.py::smoke

  # 2. Launch a detached training run (resumes from the uploaded 15k checkpoint):
  modal run --detach modal_app.py --run-id modal_qboot --steps 150000

  # 3. Pull results back to inspect locally:
  modal volume get chesszero-output runs        ./modal_out/runs
  modal volume get chesszero-output checkpoints  ./modal_out/checkpoints

  # Resuming after the 24h timeout: re-run with
  #   --resume /out/checkpoints/chess/<run-id>/checkpoint_<step>.pt
"""
import subprocess
import threading
import modal

# ── config knobs ────────────────────────────────────────────────────────────
GPU = "A100-40GB"   # 800x128 self-play peaks ~27 GB. A100 (40GB) has headroom.
                    # 24 GB cards (A10G / L4 / "L4"): drop --num-parallel-games to ~64.
                    # Override per-call: train.with_options(gpu="A100-80GB").remote(...)
APP_NAME = "chesszero"
REPO = "/root/chessZero"
PY = "/opt/venv/bin/python"   # the uv-built venv inside the image

# ── image: reproduce the repo env exactly from uv.lock ──────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("uv")
    # build deps into /opt/venv from the lockfile (exact versions, incl. torch+cu130)
    .add_local_file("pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
    .add_local_file("uv.lock", f"{REPO}/uv.lock", copy=True)
    .add_local_file("README.md", f"{REPO}/README.md", copy=True)
    .run_commands(
        f"cd {REPO} && UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-dev"
    )
    .workdir(REPO)
    # mount the source last (fast runtime layer); exclude the big / generated dirs.
    .add_local_dir(
        ".", REPO,
        ignore=[
            "data", "checkpoints", "runs", "logs", ".venv", ".git",
            "__pycache__", "*.pyc", "*.pt", "*.buf", "*.log", "*.log.*",
            ".pytest_cache", "tools/stockfish", "*.viewer*.pkl",
            "py-spy-dumps", "*.egg-info", "modal_out",
        ],
    )
)

app = modal.App(APP_NAME, image=image)

# ── volumes: data (upload once) + output (checkpoints/TB, written by training) ──
data_vol = modal.Volume.from_name("chesszero-data", create_if_missing=True)
out_vol = modal.Volume.from_name("chesszero-output", create_if_missing=True)
VOLUMES = {"/data": data_vol, "/out": out_vol}


@app.function(gpu=GPU, timeout=600)
def smoke():
    """Validate torch+cu130, triton import, and the chess_gpu kernels on a Modal GPU.
    No data volume needed — this is the cheap 'does the stack even run here' check."""
    import torch
    print("torch", torch.__version__, "| cuda avail:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
    code = (
        "import torch;"
        "from src.games.chess_gpu import GpuChessGame;"
        "g=GpuChessGame();"
        "s=g.reset_batch(16, device='cuda');"
        "lm=g.legal_mask(s);"
        "print('chess_gpu OK -> batch', s.n, 'legal_mask', tuple(lm.shape), 'on', lm.device);"
        "import src.mcts.tensor_mcts as _; print('tensor_mcts import OK')"
    )
    subprocess.run([PY, "-c", code], cwd=REPO, check=True)
    print(">>> SMOKE PASSED")


@app.function(gpu=GPU, timeout=24 * 60 * 60, volumes=VOLUMES)
def train(args: list[str]):
    """Run scripts/train.py with the given CLI args. Commits the output volume
    every 5 min so checkpoints survive the 24h timeout / preemption."""
    stop = threading.Event()

    def _committer():
        while not stop.wait(300):
            try:
                out_vol.commit()
            except Exception as e:
                print("volume commit warning:", e)

    threading.Thread(target=_committer, daemon=True).start()
    cmd = [PY, "-u", "scripts/_faulthandler_bootstrap.py", "scripts/train.py", *args]
    print("RUN:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=REPO, check=True)
    finally:
        stop.set()
        out_vol.commit()
        print(">>> output volume committed")


@app.local_entrypoint()
def main(
    run_id: str = "modal_qboot",
    steps: int = 150000,
    num_simulations: int = 800,
    num_self_play_games: int = 512,
    num_parallel_games: int = 128,
    selfplay_q_ratio: float = 0.25,
    resume: str = "/data/checkpoint_15000.pt",
    extra: str = "",
):
    """Build the train.py arg list (mirrors the local qboot_s800 command) and
    launch it on Modal. Override any knob from the CLI, e.g.:
        modal run --detach modal_app.py --run-id modal_endgame --num-parallel-games 64
    `extra` is appended verbatim (space-split) for ad-hoc flags."""
    args = [
        "--game", "chess_small",
        "--run-id", run_id,
        "--device", "cuda",
        "--steps", str(steps),
        "--eval-interval", "2000",
        "--mask-illegal-policy",
        "--stockfish-injection-path", "/data/stockfish_injection",
        "--stockfish-injection-games", "300",
        "--stockfish-injection-interval", "256",
        "--self-play-warmup-steps", "15000",
        "--warmstart-buffer-size", "300",
        "--warmstart-sample-frac", "0.4",
        "--material-value-weight", "0.5",
        "--material-value-anneal-frac", "0.6",
        "--use-material-head",
        "--material-head-loss-weight", "0.25",
        "--root-terminal-draws",
        "--root-terminal-draws-min-repeats", "2",
        "--resign-enabled",
        "--num-simulations", str(num_simulations),
        "--num-self-play-games", str(num_self_play_games),
        "--num-parallel-games", str(num_parallel_games),
        "--selfplay-q-ratio", str(selfplay_q_ratio),
        # write checkpoints + TB to the persistent output volume
        "--checkpoints-dir", "/out/checkpoints",
        "--log-dir", "/out/runs",
    ]
    if resume:
        args += ["--resume", resume]
    if extra:
        args += extra.split()
    train.remote(args)
