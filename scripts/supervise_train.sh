#!/usr/bin/env bash
# Supervise chess training: auto-restart on crash, stop on clean exit or STOP file.
#
# Exit codes:
#   0   — training completed (rc=0 from train.py)
#   130 — SIGINT (user Ctrl-C)
#   143 — SIGTERM
#   2   — supervisor misuse (missing args, wrong cwd)
#   *   — last crash's exit code after retries/no-progress cap exhausted
#
# See design in memory: supervisor_launch.md.
set -u

SCRIPT_NAME=$(basename "$0")
usage() {
  cat <<EOF
Usage: $SCRIPT_NAME --game <game> --run-id <id> [supervisor flags] [train.py args...]

Supervisor flags:
  --max-retries N        Hard cap on total attempts (default 10).
  --max-no-progress N    Cap on consecutive attempts without checkpoint advance (default 3).
  --backoff-base SECS    Base for exponential backoff (default 5).
  --backoff-cap SECS     Max backoff (default 300).
  --train-log PATH       Training stdout/stderr log (default chess_train.log).
  --ckpt-game NAME       Game dir for checkpoints/runs (default: --game value).
                         Set when config.game differs from --game — e.g. chess_small
                         saves under checkpoints/chess/, so pass --ckpt-game chess.
  --help, -h             Show this help.

All other flags pass through to scripts/train.py. The supervisor manages --resume
itself by scanning checkpoints/<game>/<run-id>/ — an explicit --resume is
honored on the first attempt only.

Create runs/<game>/<run-id>/STOP to drain a detached run without restart.
EOF
}

MAX_RETRIES=10
MAX_NO_PROGRESS=3
BACKOFF_BASE=5
BACKOFF_CAP=300
TRAIN_LOG="chess_train.log"
CKPT_GAME=""
GAME=""
RUN_ID=""
INITIAL_RESUME=""
RESET_INJECTION_CURSOR=0
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)                 usage; exit 0 ;;
    --max-retries)             MAX_RETRIES="$2"; shift 2 ;;
    --max-no-progress)         MAX_NO_PROGRESS="$2"; shift 2 ;;
    --backoff-base)            BACKOFF_BASE="$2"; shift 2 ;;
    --backoff-cap)             BACKOFF_CAP="$2"; shift 2 ;;
    --train-log)               TRAIN_LOG="$2"; shift 2 ;;
    --ckpt-game)               CKPT_GAME="$2"; shift 2 ;;
    --game)                    GAME="$2"; PASSTHROUGH+=("$1" "$2"); shift 2 ;;
    --run-id)                  RUN_ID="$2"; PASSTHROUGH+=("$1" "$2"); shift 2 ;;
    --resume)                  INITIAL_RESUME="$2"; shift 2 ;;
    # First-attempt-only: passes through with --resume on attempt 1, dropped
    # on retries. Resetting on every retry would clobber partially-advanced
    # cursors (an early-crash retry would re-inject the warmstart pool from
    # scratch, losing per-step injection progress).
    --reset-injection-cursor)  RESET_INJECTION_CURSOR=1; shift ;;
    *)                         PASSTHROUGH+=("$1"); shift ;;
  esac
done

if [[ -z "$GAME" || -z "$RUN_ID" ]]; then
  echo "ERROR: --game and --run-id are required" >&2
  usage >&2
  exit 2
fi
if [[ ! -x ".venv/bin/python" ]]; then
  echo "ERROR: .venv/bin/python not executable — run from repo root" >&2
  exit 2
fi
if [[ ! -f "scripts/train.py" ]]; then
  echo "ERROR: scripts/train.py not found — run from repo root" >&2
  exit 2
fi

CKPT_GAME="${CKPT_GAME:-$GAME}"
CHECKPOINT_DIR="checkpoints/$CKPT_GAME/$RUN_ID"
RUN_DIR="runs/$CKPT_GAME/$RUN_ID"
STOP_FILE="$RUN_DIR/STOP"
SUP_LOG="$RUN_DIR/supervisor.log"
mkdir -p "$RUN_DIR"

sup_log() {
  local line="[$(date -Iseconds)] $*"
  echo "$line" | tee -a "$SUP_LOG"
}

latest_ckpt_step() {
  ls "$CHECKPOINT_DIR"/checkpoint_*.pt 2>/dev/null |
    sed -n 's|.*/checkpoint_\([0-9]\+\)\.pt|\1|p' |
    sort -n | tail -1
}

retry=0
no_progress=0
prev_step=""

sup_log "supervisor start: game=$GAME run-id=$RUN_ID max-retries=$MAX_RETRIES max-no-progress=$MAX_NO_PROGRESS"

while true; do
  if [[ -f "$STOP_FILE" ]]; then
    sup_log "STOP file present at $STOP_FILE — exit without restart"
    exit 0
  fi

  step=$(latest_ckpt_step)
  resume_args=()
  if [[ $retry -eq 0 && -n "$INITIAL_RESUME" ]]; then
    resume_args=(--resume "$INITIAL_RESUME")
    if [[ $RESET_INJECTION_CURSOR -eq 1 ]]; then
      resume_args+=(--reset-injection-cursor)
      sup_log "attempt $((retry+1)): user-supplied --resume $INITIAL_RESUME (with --reset-injection-cursor)"
    else
      sup_log "attempt $((retry+1)): user-supplied --resume $INITIAL_RESUME"
    fi
  elif [[ -n "$step" ]]; then
    resume_args=(--resume "$CHECKPOINT_DIR/checkpoint_${step}.pt")
    sup_log "attempt $((retry+1)): resume from step $step"
  else
    sup_log "attempt $((retry+1)): cold start (no checkpoint)"
  fi

  if [[ $retry -gt 0 ]]; then
    if [[ "$step" == "$prev_step" ]]; then
      no_progress=$((no_progress+1))
      sup_log "no forward progress since last attempt ($no_progress/$MAX_NO_PROGRESS)"
    else
      no_progress=0
    fi
  fi
  prev_step="$step"

  {
    echo ""
    echo "=== supervisor attempt $((retry+1)) at $(date -Iseconds) ==="
  } | tee -a "$TRAIN_LOG"

  start_ts=$(date +%s)
  # ``python -u``: unbuffered stdout/stderr. Without it, stdout is fully-
  # buffered when piped through tee (4 KB buffer), so ``tqdm.write`` from
  # the inner self-play loops doesn't appear for many minutes. The tqdm
  # progress bar itself uses stderr (line-buffered) and shows up fine.
  .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
    "${PASSTHROUGH[@]}" "${resume_args[@]}" 2>&1 | tee -a "$TRAIN_LOG"
  rc=${PIPESTATUS[0]}
  elapsed=$(($(date +%s) - start_ts))

  sup_log "training exited rc=$rc after ${elapsed}s"

  case $rc in
    0)        sup_log "clean exit — training complete"; exit 0 ;;
    130|143)  sup_log "interrupted (rc=$rc) — not restarting"; exit $rc ;;
  esac

  {
    echo "--- dmesg tail ---"
    dmesg -T 2>/dev/null | tail -20 || echo "(dmesg not readable — skip sudo sysctl kernel.dmesg_restrict=0 if needed)"
    echo "--- free -h ---"
    free -h 2>/dev/null || true
    echo "--- end crash context ---"
  } >> "$SUP_LOG"

  retry=$((retry+1))
  if [[ $retry -ge $MAX_RETRIES ]]; then
    sup_log "max-retries ($MAX_RETRIES) exhausted — giving up (rc=$rc)"
    exit $rc
  fi
  if [[ $no_progress -ge $MAX_NO_PROGRESS ]]; then
    sup_log "max-no-progress ($MAX_NO_PROGRESS) stuck attempts — giving up (rc=$rc)"
    exit $rc
  fi

  sleep_s=$((BACKOFF_BASE * (1 << retry)))
  [[ $sleep_s -gt $BACKOFF_CAP ]] && sleep_s=$BACKOFF_CAP
  sup_log "backoff ${sleep_s}s before restart"
  sleep "$sleep_s"
done
