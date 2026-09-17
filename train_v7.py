#!/usr/bin/env python3
"""Launch model_v7 (V7 line) on this laptop's RTX PRO 1000 Blackwell (8 GB, sm_120).

Use ./run.sh, which sources env.sh first — TensorFlow silently falls back to
the CPU without the library path that script exports.

    ./run.sh selftest     ~1 min, no dataset or GPU needed
    ./run.sh prepare      one-off: PNG + YOLO polygons -> memmapped arrays
    ./run.sh benchmark    measure step time and VRAM per batch size
    ./run.sh scratch      train from random initialisation
    ./run.sh report       print progress; safe while training runs

Everything is scratch initialisation: no ImageNet weights, no V5 warm start,
nothing downloaded. `transfer` is deliberately not offered here.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("selftest", "prepare", "benchmark", "scratch", "report", "_bench_one")


def load_model_module():
    path = HERE / "model_v7.py"
    if not path.is_file():
        sys.exit(f"{path} is not next to this script.")
    spec = importlib.util.spec_from_file_location("model_v7", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_v7"] = module
    spec.loader.exec_module(module)
    return module


def apply_laptop_configuration(m, *, require_gpu: bool) -> None:
    """The measured configuration for this machine.

    Every value here was chosen from a measurement in this repository's
    benchmark mode or from the dataset audit, not from a default.
    """
    m.REQUIRE_GPU = require_gpu
    m.USE_MIXED_PRECISION = True

    # Measured on this machine (see logs/fitbench.log and logs/ab.log):
    #
    #   batch 2                 752 ms/step, 3.7 GiB peak
    #   batch 4, 6, 8           OOM -- Keras GroupNormalization materialises
    #                           [N, groups, HW, C/groups] moments and 8 GB
    #                           cannot hold them past batch 2
    #   steps_per_execution 8   -2.5%
    #   per-class IoU off       -1.3%
    #   accumulation 2          -4.2% per image, and it doubles the effective
    #                           batch: fewer AdamW+EMA applies over 6.4M
    #                           parameters more than pays for the extra step
    #   XLA                     does not compile: ResizeBilinearGrad mixes
    #                           f32 and f16 under mixed_float16
    #
    # The model uses GroupNormalization, not BatchNorm, so a batch of 2 costs
    # nothing statistically -- normalisation here is per-sample either way.
    m.BATCH_SIZE = 2
    m.GRADIENT_ACCUMULATION_STEPS = 2
    m.STEPS_PER_EXECUTION = 8
    m.PER_CLASS_IOU_METRICS_DURING_TRAINING = False
    m.USE_XLA_JIT = False

    # One loader thread would keep up: 23 ms/image against a 376 ms/image
    # step. Six is margin for the slowest augmentation modes without taking
    # cores from TensorFlow's own op threads.
    m.DATA_LOADER_WORKERS = 6
    m.DATA_LOADER_QUEUE_SIZE = 16
    m.PREPARE_WORKERS = 16

    # 29.3 min/epoch of training steps measured, plus validation over the
    # full 1,048-image split and an instance evaluation every 5 epochs:
    # roughly 33 min/epoch, so 120 epochs is about 2.7 days. The cosine
    # schedule anneals over exactly EPOCHS, so this cannot be shortened
    # mid-run without leaving the model un-annealed, and it must exceed
    # AUGMENTATION_CYCLE_LENGTH (95) for every image to see every V5 mode.
    m.EPOCHS = 120

    if out := os.environ.get("PCB_MODEL_OUTPUT_DIR"):
        m.MODEL_OUTPUT_DIR = Path(out)

    # Overridable from the environment so benchmark subprocesses can sweep them.
    for name, cast in (
        ("PCB_BATCH_SIZE", int),
        ("PCB_GRADIENT_ACCUMULATION_STEPS", int),
        ("PCB_EPOCHS", int),
        ("PCB_STEPS_PER_EXECUTION", int),
        ("PCB_LEARNING_RATE", float),
        ("PCB_USE_XLA_JIT", lambda v: v.strip().lower() in ("1", "true", "yes")),
        ("PCB_PER_CLASS_IOU_METRICS_DURING_TRAINING",
         lambda v: v.strip().lower() in ("1", "true", "yes")),
    ):
        raw = os.environ.get(name)
        if raw:
            setattr(m, name[len("PCB_"):], cast(raw))


def describe(m, epochs: int) -> None:
    line = "=" * 74
    print(line)
    print("Model_v7 — scratch (random initialisation, no pretrained weights)")
    print(f"  dataset   : {m.DATASET_ROOT}")
    print(f"  arrays    : {m.ARRAY_DIR}")
    print(f"  output    : {m.MODEL_OUTPUT_DIR}")
    print(f"  input     : {m.IMG_SIZE}px letterboxed, "
          f"instance head {m.INSTANCE_HEAD_SIZE}px")
    print(f"  batch     : {m.BATCH_SIZE} x {m.GRADIENT_ACCUMULATION_STEPS} "
          f"accumulation (effective {m.BATCH_SIZE * m.GRADIENT_ACCUMULATION_STEPS})")
    print(f"  steps/exec: {m.STEPS_PER_EXECUTION}")
    print(f"  precision : "
          f"{'mixed_float16' if m.USE_MIXED_PRECISION else 'float32'}")
    print(f"  epochs    : {epochs}   lr {m.LEARNING_RATE} -> "
          f"{m.MIN_LEARNING_RATE} (cosine, {m.WARMUP_EPOCHS} warmup)")
    print(f"  loaders   : {m.DATA_LOADER_WORKERS} threads, "
          f"queue {m.DATA_LOADER_QUEUE_SIZE}")
    print(f"  selection : {m.INSTANCE_SELECTION_METRIC} on validation")
    print(f"  report    : {m.MODEL_OUTPUT_DIR / 'TRAINING_REPORT.md'}")
    print(line, flush=True)


# --------------------------------------------------------------- benchmarking
def benchmark_one(m, batch_size: int, steps: int = 12) -> dict:
    """Time real training steps at one batch size and report peak VRAM."""
    import numpy as np
    import tensorflow as tf

    m.BATCH_SIZE = batch_size
    m.GRADIENT_ACCUMULATION_STEPS = 1
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    m.configure_runtime()

    arrays = m.load_split_arrays("train")
    sequence = m.InstanceArraySequence(
        arrays, batch_size=batch_size, training=True
    )

    # Data loader alone, so a slow step can be attributed.
    t0 = time.perf_counter()
    for index in range(4):
        sequence[index]
    loader_seconds = (time.perf_counter() - t0) / (4 * batch_size)

    model = m.build_model_v7_instance()
    weights = m.compute_class_weights(arrays["semantic"])
    m.compile_model(model, weights, learning_rate=m.LEARNING_RATE)

    tf.config.experimental.reset_memory_stats("GPU:0")
    batches = [sequence[i] for i in range(steps + 3)]
    for index in range(3):  # warm up: PTX JIT, cuDNN autotune, XLA if enabled
        model.train_on_batch(*batches[index])

    t0 = time.perf_counter()
    for index in range(3, steps + 3):
        model.train_on_batch(*batches[index])
    step_seconds = (time.perf_counter() - t0) / steps

    info = tf.config.experimental.get_memory_info("GPU:0")
    train_images = len(arrays["images"])
    return {
        "batch_size": batch_size,
        "step_seconds": step_seconds,
        "seconds_per_image": step_seconds / batch_size,
        "loader_seconds_per_image": loader_seconds,
        "peak_gib": info["peak"] / (1024 ** 3),
        "epoch_minutes": step_seconds * (train_images / batch_size) / 60.0,
        "train_images": train_images,
    }


def run_benchmark() -> None:
    """Sweep batch sizes, each in its own process so an OOM is survivable."""
    candidates = [int(x) for x in
                  os.environ.get("PCB_BENCH_BATCHES", "2,4,6,8").split(",")]
    results = []
    for batch_size in candidates:
        print(f"\n--- batch {batch_size} ---", flush=True)
        environment = dict(os.environ, PCB_BENCH_BATCH=str(batch_size))
        completed = subprocess.run(
            [sys.executable, "-u", str(HERE / "train_v7.py"), "_bench_one"],
            env=environment, capture_output=True, text=True,
        )
        payload = None
        for line in completed.stdout.splitlines():
            if line.startswith("BENCH_JSON "):
                payload = json.loads(line[len("BENCH_JSON "):])
        if payload is None:
            tail = (completed.stdout + completed.stderr).strip().splitlines()
            reason = "OOM" if any("OOM" in l or "ResourceExhausted" in l
                                  for l in tail) else "failed"
            print(f"  {reason}: " + "\n         ".join(tail[-4:]))
            results.append({"batch_size": batch_size, "failed": reason})
            continue
        print(f"  {payload['step_seconds']*1000:7.0f} ms/step  "
              f"{payload['seconds_per_image']*1000:6.0f} ms/image  "
              f"peak {payload['peak_gib']:.2f} GiB  "
              f"{payload['epoch_minutes']:.1f} min/epoch")
        results.append(payload)

    print("\n" + "=" * 74)
    print(f"{'batch':>6} {'ms/step':>9} {'ms/image':>9} {'loader ms':>10} "
          f"{'peak GiB':>9} {'min/epoch':>10}")
    for row in results:
        if row.get("failed"):
            print(f"{row['batch_size']:>6} {row['failed']:>9}")
            continue
        print(f"{row['batch_size']:>6} {row['step_seconds']*1000:>9.0f} "
              f"{row['seconds_per_image']*1000:>9.0f} "
              f"{row['loader_seconds_per_image']*1000:>10.1f} "
              f"{row['peak_gib']:>9.2f} {row['epoch_minutes']:>10.1f}")
    print("=" * 74)
    good = [r for r in results if not r.get("failed")]
    if good:
        best = min(good, key=lambda r: r["seconds_per_image"])
        print(f"fastest per image: batch {best['batch_size']} "
              f"({best['epoch_minutes']:.1f} min/epoch, "
              f"peak {best['peak_gib']:.2f} GiB)")
    (HERE / "benchmark_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


def main() -> None:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "report").strip().lower()
    if mode not in MODES:
        sys.exit(f"usage: python train_v7.py [{'|'.join(MODES[:-1])}]")

    if mode == "benchmark":
        run_benchmark()
        return

    m = load_model_module()
    apply_laptop_configuration(
        m, require_gpu=mode in ("scratch", "benchmark", "_bench_one")
    )
    m.validate_spatial_configuration()
    m.validate_augmentation_configuration()

    if mode == "_bench_one":
        payload = benchmark_one(m, int(os.environ["PCB_BENCH_BATCH"]))
        print("BENCH_JSON " + json.dumps(payload))
        return

    if mode == "selftest":
        m.REQUIRE_GPU = False
        m.run_selftest()
        return

    if mode == "prepare":
        started = time.time()
        m.prepare_dataset()
        print(f"prepare took {(time.time() - started) / 60:.1f} min")
        m.save_dataset_preview()
        return

    if mode == "report":
        if (m.MODEL_OUTPUT_DIR / "training_log.csv").is_file():
            print(m.build_training_report(m.MODEL_OUTPUT_DIR))
            m.write_training_report(m.MODEL_OUTPUT_DIR)
            return
        sys.exit(f"No training_log.csv in {m.MODEL_OUTPUT_DIR}.")

    describe(m, m.EPOCHS)
    m.train(fine_tune=False, transfer=False)


if __name__ == "__main__":
    main()
