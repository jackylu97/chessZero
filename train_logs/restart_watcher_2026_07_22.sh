#!/bin/bash
# 2026-07-22 veto-fix bundle restart watcher (quiescence-safe — see crash #4,
# 2026-07-20: NEVER signal the trainer right after a .buf appears; the ~270MB
# pickle write is not atomic).
#
# Waits for a .buf with step >= MIN_STEP, requires size stable across 60s AND
# mtime age > 120s, then C-c's the production tmux pane, waits for the trainer
# to exit, and relaunches via launch_ft2_reanalyze.sh (which resolves the
# newest .pt+.buf pair itself). Prints progress lines; exits after relaunch.
set -u
RUN_DIR=/home/user/chessZero/checkpoints/chess/2026_07_08_prod_xl_ft_s800
MIN_STEP=${MIN_STEP:-104000}
PANE=production

newest_buf_step() {
  ls -1 "$RUN_DIR"/checkpoint_*.buf 2>/dev/null \
    | sed -E 's/.*checkpoint_([0-9]+)\.buf/\1/' | sort -n | tail -1
}

echo "watcher armed: waiting for .buf step >= $MIN_STEP"
while true; do
  step=$(newest_buf_step)
  if [ -n "$step" ] && [ "$step" -ge "$MIN_STEP" ]; then
    buf="$RUN_DIR/checkpoint_${step}.buf"
    s1=$(stat -c %s "$buf"); sleep 60; s2=$(stat -c %s "$buf")
    age=$(( $(date +%s) - $(stat -c %Y "$buf") ))
    if [ "$s1" = "$s2" ] && [ "$age" -gt 120 ] && [ -f "$RUN_DIR/checkpoint_${step}.pt" ]; then
      echo "quiescent .buf at step $step (size $s2, age ${age}s) — stopping trainer"
      tmux send-keys -t "$PANE" C-c
      # Wait for the trainer process to exit (SIGINT triggers a clean shutdown).
      for i in $(seq 1 120); do
        pgrep -f "_faulthandler""_bootstrap" >/dev/null 2>&1 || break
        sleep 5
      done
      if pgrep -f "_faulthandler""_bootstrap" >/dev/null 2>&1; then
        echo "WARN: trainer still alive after 10 min — second C-c"
        tmux send-keys -t "$PANE" C-c
        sleep 60
      fi
      echo "relaunching via launch_ft2_reanalyze.sh"
      tmux send-keys -t "$PANE" "bash train_logs/launch_ft2_reanalyze.sh" Enter
      # Buffer load replays 10k games via from_compact_dict — takes minutes.
      sleep 600
      echo "=== post-restart check (must show 'Loaded replay buffer (N games)') ==="
      grep -a "Resuming from" /home/user/chessZero/selfplay_production.log | tail -1
      grep -a "Loaded replay buffer" /home/user/chessZero/selfplay_production.log | tail -1
      grep -aE "corrupt|empty buffer|Traceback|FATAL" /home/user/chessZero/selfplay_production.log | tail -3
      exit 0
    fi
    echo "step $step present but not quiescent yet (size $s1->$s2, age ${age}s)"
  fi
  sleep 60
done
