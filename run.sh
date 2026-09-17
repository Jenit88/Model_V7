#!/bin/bash
# Single entry point for every model_v7 run mode on this laptop.
#
#   ./run.sh selftest    ~1 min, no dataset or GPU needed
#   ./run.sh prepare     one-off: PNG + YOLO polygons -> memmapped arrays
#   ./run.sh benchmark   measure step time / VRAM at several batch sizes
#   ./run.sh scratch     train from random initialisation
#   ./run.sh report      print progress; safe to run while training
#   ./run.sh dashboard   serve the live dashboard on :8088
#   ./run.sh tensorboard serve TensorBoard on :6006
#
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

MODE="${1:-report}"
shift || true

case "$MODE" in
  selftest|prepare|benchmark|scratch|report)
    exec "$PCB_PYTHON" -u train_v7.py "$MODE" "$@"
    ;;
  dashboard)
    exec "$PCB_PYTHON" -u dashboard.py "$PCB_MODEL_OUTPUT_DIR" "$@"
    ;;
  predict-folder)
    # predict_folder.py's own usage line advertised this mode; run.sh never
    # defined it, so the documented command failed.
    exec "$PCB_PYTHON" -u predict_folder.py "$@"
    ;;
  measure-fallback)
    exec "$PCB_PYTHON" -u measure_fallback.py "$@"
    ;;
  reevaluate)
    exec "$PCB_PYTHON" -u reevaluate.py "$@"
    ;;
  fit-thresholds)
    exec "$PCB_PYTHON" -u fit_thresholds.py "$@"
    ;;
  tta-selfcheck)
    exec "$PCB_PYTHON" -u tta.py "$@"
    ;;
  tensorboard)
    # Serve the run directory, not only its tensorboard/ subdirectory: Keras'
    # callback writes just batch_* scalars there (its validation event file is a
    # 78-byte header), so every epoch, validation and instance curve comes from
    # tb_export.py writing a sibling tensorboard_metrics/. The parent shows both.
    #
    # --bind_all so WSL2's localhost forwarding reaches it from Windows;
    # reload_interval 15 so a long run's curves keep moving without hammering
    # the event files the training process is still appending to.
    exec "$PCB_PYTHON" -u -m tensorboard.main \
      --logdir "$PCB_MODEL_OUTPUT_DIR" \
      --port 6006 --bind_all --reload_interval 15 \
      --reload_multifile true "$@"
    ;;
  *)
    echo "usage: $0 {selftest|prepare|benchmark|scratch|report|dashboard|tensorboard|predict-folder|fit-thresholds|reevaluate|tta-selfcheck}" >&2
    exit 2
    ;;
esac
