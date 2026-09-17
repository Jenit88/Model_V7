#!/usr/bin/env python
"""Predict a folder of images and keep only the instance overlay PNGs.

model_v7.predict_one_image writes eight artefacts per image (two .npy, five
.png, one .json). This runs the same code path -- so letterboxing, decoding and
geometry restoration are identical -- but writes each image's artefacts into one
reusable scratch directory and keeps only ``instances.png``, renamed to the
source image's own name.

    ./env.sh && $PCB_PYTHON predict_pictures.py <source_dir> <output_dir>

Resumable: an image whose output already exists is skipped.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import importlib.util as _importlib_util

# Load model_v7.py under the module name the rest of the project uses. A plain
# `import model_v7` would need a file of that name; this keeps one model file.
_spec = _importlib_util.spec_from_file_location(
    "model_v7", Path(__file__).resolve().parent / "model_v7.py"
)
m = _importlib_util.module_from_spec(_spec)
sys.modules["model_v7"] = m
_spec.loader.exec_module(m)
import tensorflow as tf

from deployment import apply_deployment_profile

# model_v7's module-level MODEL_FOR_INFERENCE points at the transfer run.
# This is the scratch run, so take the checkpoint that final_model_selection.json
# names as authoritative, overridable from the environment.
DEFAULT_MODEL = (
    Path(os.environ.get("PCB_MODEL_OUTPUT_DIR", "~/Models/Model_v7_scratch_rtx"))
    .expanduser()
    / "best_model_v7_instance.keras"
)


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <source_dir> <output_dir>")

    source = Path(sys.argv[1]).expanduser().resolve(strict=False)
    out_dir = Path(sys.argv[2]).expanduser().resolve(strict=False)
    if not source.is_dir():
        sys.exit(f"source is not a directory: {source}")

    m.MODEL_FOR_INFERENCE = Path(
        os.environ.get("PCB_MODEL_FOR_INFERENCE", DEFAULT_MODEL)
    )
    model_path = Path(m.MODEL_FOR_INFERENCE).expanduser().resolve(strict=False)
    if not model_path.is_file():
        sys.exit(f"inference model not found: {model_path}")

    # Real boards, so the deployment operating point rather than the benchmark
    # one. See deployment.py for the measured difference between them.
    apply_deployment_profile(m)

    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_scratch"

    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if m.USE_MIXED_PRECISION else "float32"
    )
    m.configure_runtime()

    images = m.find_prediction_images(source, excluded_roots=(out_dir,))
    print(f"model  : {model_path}", flush=True)
    print(f"source : {source}", flush=True)
    print(f"output : {out_dir}", flush=True)
    print(f"images : {len(images)}", flush=True)

    model = tf.keras.models.load_model(model_path, compile=False)
    m.validate_model_output_shapes(model)
    m.unpack_model_outputs(
        model(tf.zeros((1, m.IMG_SIZE, m.IMG_SIZE, 3), tf.float32), training=False)
    )
    print("warm-up complete", flush=True)

    done = skipped = failed = 0
    run_start = time.perf_counter()

    for index, image_path in enumerate(images, start=1):
        target = out_dir / f"{image_path.stem}.png"
        if target.exists():
            skipped += 1
            continue

        try:
            if scratch.exists():
                shutil.rmtree(scratch)
            scratch.mkdir(parents=True)

            m.predict_one_image(model, image_path, scratch, model_path=model_path)

            overlay = scratch / "instances.png"
            if not overlay.is_file():
                raise FileNotFoundError("instances.png was not produced")
            shutil.move(str(overlay), str(target))
            done += 1
        except Exception as error:  # keep going; one bad image must not stop the run
            failed += 1
            print(f"FAILED {image_path.name}: {error}", flush=True)

        if index % 25 == 0 or index == len(images):
            elapsed = time.perf_counter() - run_start
            rate = done / elapsed if elapsed > 0 and done else 0.0
            remaining = (len(images) - index) / rate / 60 if rate else 0.0
            print(
                f"[{index}/{len(images)}] written={done} skipped={skipped} "
                f"failed={failed} {rate:.2f} img/s eta={remaining:.0f} min",
                flush=True,
            )

    if scratch.exists():
        shutil.rmtree(scratch)

    print(
        f"\ndone. written={done} skipped={skipped} failed={failed} "
        f"in {(time.perf_counter() - run_start) / 60:.1f} min",
        flush=True,
    )
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
