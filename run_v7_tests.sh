#!/bin/bash
# Every V7 contract, on CPU so it never contends with a training run.
#
#   ./run_v7_tests.sh
#
# CPU-only is deliberate rather than a limitation: these are decode, target and
# geometry contracts, none of which depend on the accelerator, and the GPU is
# committed to a multi-day run this must not disturb.
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=""
PYTHON="${PCB_PYTHON:-$HOME/envs/pcb62/bin/python}"

FAILED=0
for name in run_selftest_v7 test_tier2_grouping test_tier2_loss test_tier3_tiling test_v7_decode test_v7_predict; do
  printf '%-26s ' "$name"
  if "$PYTHON" -u "$name.py" > "/tmp/$name.out" 2>&1; then
    echo "PASS"
  else
    echo "FAIL   (see /tmp/$name.out)"
    tail -5 "/tmp/$name.out" | sed 's/^/      /'
    FAILED=$((FAILED + 1))
  fi
done

echo
if [ "$FAILED" -eq 0 ]; then
  echo "all V7 contracts hold"
else
  echo "$FAILED suite(s) failed"
  exit 1
fi
