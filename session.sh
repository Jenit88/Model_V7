#!/bin/bash
# Create (or reuse) the detached tmux session that holds every long-running
# job, so nothing dies when a shell or a VS Code window closes, and so the run
# can be watched live.
#
#   ./session.sh start <mode>   run ./run.sh <mode> in the session's work window
#   ./session.sh attach         watch it  (detach again with Ctrl-b then d)
#   ./session.sh status         one-screen summary without attaching
#   ./session.sh stop           kill the session and everything in it
#
# Windows in the session:
#   work  the training / prepare / benchmark job
#   dash  the live dashboard on http://localhost:8088
#   tb    TensorBoard on http://localhost:6006
#   gpu   nvidia-smi, refreshed every 2 s
set -uo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
SESSION="${PCB_TMUX_SESSION:-v7}"
# tmux resolves a target session by *prefix*, so `-t v7` also matches the
# supervisor's own `v7sup` session (created by the Windows logon task). On
# 10-09 that made the resume a no-op: has-session matched `v7sup` and returned
# early, so the `work` window was never created, `respawn-pane -t pcb:work`
# failed with "can't find window: work", and only the aux windows came up --
# against the default output dir. `=` forces an exact name match.
SESSION_T="=$SESSION"

# Built once, at the top, because the *aux* windows need it as much as the work
# pane does. tb and dash call ./run.sh with no prefix if this is left to the
# `start` case, so they fall back to env.sh's default output dir -- on 10-09
# TensorBoard and the dashboard spent a resume reporting the run that finished
# on 05-09 while training ran against Tier-1.
ENV_PREFIX=""
# The three paths, plus the knobs a short experimental run needs. PCB_EPOCHS was
# missing, so `PCB_EPOCHS=1 ./session.sh start scratch` silently started a full
# 120-epoch run -- the launcher accepted the variable and the pane never saw it.
# Short runs are how the loss weight gets swept without spending 2.8 days a time,
# so this list has to carry them.
for name in PCB_ARRAY_DIR PCB_MODEL_OUTPUT_DIR PCB_DATASET_ROOT             PCB_EPOCHS PCB_BATCH_SIZE PCB_LEARNING_RATE             PCB_GRADIENT_ACCUMULATION_STEPS PCB_STEPS_PER_EXECUTION             PCB_USE_XLA_JIT PCB_BENCH_BATCHES; do
  value="$(eval "printf '%s' \"\${$name-}\"")"
  [ -n "$value" ] && ENV_PREFIX="$ENV_PREFIX $name='$value'"
done

start_session() {
  tmux has-session -t "$SESSION_T" 2>/dev/null && return 0
  tmux new-session -d -s "$SESSION" -n work -c "$HERE"
  tmux new-window  -t "$SESSION_T" -n dash -c "$HERE" \
    "while true; do env$ENV_PREFIX ./run.sh dashboard; echo 'dashboard exited; restarting in 5s'; sleep 5; done"
  tmux new-window  -t "$SESSION_T" -n tb -c "$HERE" \
    "while true; do env$ENV_PREFIX ./run.sh tensorboard; echo 'tensorboard exited; restarting in 5s'; sleep 5; done"
  tmux new-window  -t "$SESSION_T" -n gpu \
    "watch -n2 nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm --format=csv"
  tmux select-window -t "$SESSION_T:work"
}

# Bring up any window the session is missing, so an already-running session
# picks up a newly added one without being torn down.
ensure_windows() {
  tmux has-session -t "$SESSION_T" 2>/dev/null || return 0
  tmux list-windows -t "$SESSION_T" -F '#{window_name}' | grep -qx tb || \
    tmux new-window -d -t "$SESSION_T" -n tb -c "$HERE" \
      "while true; do env$ENV_PREFIX ./run.sh tensorboard; echo 'tensorboard exited; restarting in 5s'; sleep 5; done"
  tmux list-windows -t "$SESSION_T" -F '#{window_name}' | grep -qx dash || \
    tmux new-window -d -t "$SESSION_T" -n dash -c "$HERE" \
      "while true; do env$ENV_PREFIX ./run.sh dashboard; echo 'dashboard exited; restarting in 5s'; sleep 5; done"
}

case "${1:-status}" in
  start)
    MODE="${2:?usage: ./session.sh start <selftest|prepare|benchmark|scratch>}"
    start_session
    ensure_windows
    # Forward the path overrides explicitly. A pane inherits the *tmux
    # server's* environment, captured whenever that server first started --
    # which may be days ago, for an unrelated session. Exporting a variable in
    # the shell that calls this script does not reach the pane, so a run
    # launched against a different array or output directory silently used the
    # old one. On 08-09 that sent a Tier-1 scratch run at the finished run's
    # output directory; train()'s checkpoint guard caught it, but only because
    # that guard exists.
    [ -n "$ENV_PREFIX" ] && echo "  overrides   :$ENV_PREFIX"
    # respawn-pane replaces whatever was in the work window, so re-running
    # start never stacks two training jobs on the same GPU.
    tmux respawn-pane -k -t "$SESSION_T:work" -c "$HERE" \
      "env$ENV_PREFIX ./run.sh $MODE 2>&1 | tee -a '$HERE/logs/$MODE.log'; \
       echo; echo '--- $MODE finished, exit '\$?' --- (window stays open)'; \
       exec bash"
    echo "started '$MODE' in tmux session '$SESSION'"
    echo "  watch       : ./session.sh attach     (detach: Ctrl-b then d)"
    echo "  dashboard   : http://localhost:8088"
    echo "  tensorboard : http://localhost:6006"
    echo "  log         : $HERE/logs/$MODE.log"
    ;;
  attach)  ensure_windows; exec tmux attach -t "$SESSION_T" ;;
  serve)   ensure_windows; echo "dashboard :8088   tensorboard :6006" ;;
  status)
    if ! tmux has-session -t "$SESSION_T" 2>/dev/null; then
      echo "no tmux session '$SESSION'"; exit 1
    fi
    tmux list-windows -t "$SESSION_T" -F "#{window_name}: #{pane_current_command}"
    echo
    tmux capture-pane -p -t "$SESSION_T:work" | grep -v '^$' | tail -20
    ;;
  stop)    tmux kill-session -t "$SESSION_T" 2>/dev/null; echo "stopped" ;;
  *)       echo "usage: $0 {start <mode>|attach|serve|status|stop}" >&2; exit 2 ;;
esac
