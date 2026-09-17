#!/usr/bin/env python3
"""Run the finished model over a random sample of a folder of real images.

Unlike `predict()` this makes no assumption that the images resemble the
training set -- these are whatever is on disk -- so it reports the detection
score distribution alongside the pictures. That distribution is the honest
signal on unlabelled data: with no ground truth there is no precision to quote,
but a healthy run puts detections at 0.8-1.0 and leaves the 0.1-0.3 band empty.

    PCB_SOURCE="/mnt/c/.../_Pictures" PCB_SAMPLE=40 ./run.sh predict-folder
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from deployment import apply_deployment_profile

HERE = Path(__file__).resolve().parent


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "model_v7", HERE / "model_v7.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_v7"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    m = load_model_module()
    import tensorflow as tf

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    m.REQUIRE_GPU = False

    source = Path(os.environ["PCB_SOURCE"])
    sample = int(os.environ.get("PCB_SAMPLE", "40"))
    destination = Path(os.environ["PCB_PREDICT_OUT"])
    seed = int(os.environ.get("PCB_SAMPLE_SEED", "20260905"))

    m.MODEL_FOR_INFERENCE = (
        Path(os.environ["PCB_MODEL_OUTPUT_DIR"]) / "best_model_v7_instance.keras"
    )
    m.USE_TTA_AT_EVALUATION = False   # predict() runs its own inference path

    # This folder is real boards, not the labelled splits, so it runs at the
    # deployment operating point. See deployment.py for why that differs from
    # the benchmark one, and what it was measured to do.
    apply_deployment_profile(m)

    candidates = sorted(
        p for p in source.rglob("*")
        if p.suffix.lower() in m.IMAGE_EXTENSIONS and p.is_file()
    )
    print(f"source     : {source}")
    print(f"available  : {len(candidates):,} images")
    if not candidates:
        sys.exit("no images found")

    random.Random(seed).shuffle(candidates)
    chosen = candidates[:sample]
    print(f"sampling   : {len(chosen)} at random (seed {seed})")
    print(f"model      : {m.MODEL_FOR_INFERENCE.name}\n", flush=True)

    staging = Path("/tmp/pcb_folder_sample")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for index, path in enumerate(chosen):
        shutil.copy2(path, staging / f"{index:03d}_{path.name}")

    m.PREDICT_SOURCE = staging
    destination.mkdir(parents=True, exist_ok=True)
    # predict() appends into its output root, so a previous run's folders are
    # still there and would be counted as if they belonged to this one.
    predictions_root = Path(os.environ["PCB_MODEL_OUTPUT_DIR"]) / "predictions"
    shutil.rmtree(predictions_root, ignore_errors=True)
    started = time.time()
    result = m.predict()

    # Move the rendered overlays somewhere findable from Windows, flat.
    # Never mix two runs' pictures. rmtree can leave the directory itself
    # behind on the 9p /mnt/c mount, so clear the files and tolerate the dir.
    overlays = destination / "overlays"
    if overlays.is_dir():
        for stale in overlays.glob("*.png"):
            stale.unlink(missing_ok=True)
    overlays.mkdir(parents=True, exist_ok=True)
    root = Path(result["output_root"])
    scores, per_class, per_image = [], defaultdict(int), []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        overlay = folder / "instances.png"
        if overlay.is_file():
            shutil.copy2(overlay, overlays / f"{folder.name[:60]}.png")
        meta = folder / "instances.json"
        if meta.is_file():
            data = json.loads(meta.read_text())
            instances = data.get("instances", [])
            per_image.append((folder.name, len(instances)))
            for item in instances:
                scores.append(float(item["detection_score"]))
                per_class[item["class_name"]] += 1

    elapsed = time.time() - started
    print(f"\n{'=' * 66}")
    print(f"{len(chosen)} unseen images, {len(scores):,} detections, "
          f"{elapsed / max(len(chosen), 1):.2f} s/image")
    print(f"objects per image: min {min(n for _, n in per_image)}, "
          f"median {sorted(n for _, n in per_image)[len(per_image) // 2]}, "
          f"max {max(n for _, n in per_image)}")
    print("\nby class:")
    for name, count in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20}{count:>7,}  {count / max(len(scores),1):>6.1%}")
    print("\ndetection-score distribution (no labels here, so this is the "
          "\nhonest health signal -- a good run leaves the low bands empty):")
    for low, high in ((0.0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        n = sum(1 for s in scores if low <= s < high)
        bar = "#" * int(round(46 * n / max(len(scores), 1)))
        print(f"  {low:.1f}-{high:<4.1f}{n:>7,}  {n / max(len(scores),1):>6.1%} {bar}")
    print(f"{'=' * 66}")
    print(f"overlays : {overlays}")

    (destination / "summary.json").write_text(json.dumps({
        "source": str(source), "sampled": len(chosen),
        "detections": len(scores), "per_class": dict(per_class),
        "seconds_per_image": elapsed / max(len(chosen), 1),
        "score_bands": {f"{a}-{b}": sum(1 for s in scores if a <= s < b)
                        for a, b in ((0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.01))},
        "per_image": per_image,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
