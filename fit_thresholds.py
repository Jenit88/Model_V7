#!/usr/bin/env python3
"""Fit the decoder thresholds on validation, honestly.

The five thresholds in `decode_instances` are still the ones Model V5 was tuned
with, and they are shared across all four classes. This searches them.

Two disciplines make the result mean something, and both are the point of the
script rather than decoration:

**Split validation.** The validation set is halved by image index. Candidates
are scored on the fitting half and the winner is re-scored on the held-out
half. A candidate that wins on the fitting half and collapses on the other was
fitting noise, and the printed table shows that happening rather than hiding it.

**Cached predictions.** The model runs once per image; every candidate is then
scored by re-decoding cached probability maps. Otherwise 200 candidates would
mean 200 forward passes over the split.

A caveat that has to travel with any number this produces: 44 images are
byte-identical between validation and test, so thresholds fitted here
contaminate roughly 10% of the test split. Say so wherever the test number is
quoted.

    PCB_FIT_IMAGES=300 PCB_FIT_DRAWS=120 ./run.sh fit-thresholds
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "model_v7", HERE / "model_v7.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_v7"] = module
    spec.loader.exec_module(module)
    return module


# name -> (attribute on the module, sampler)
#
# This is V7's decoder vocabulary, not V6's. `center_confidence` and
# `center_nms_radius` were still being searched here after the centre head was
# deleted: they set attributes the distance decoder never reads, so half the
# search dimensions were inert and every reported gain came from the other half.
# The two new entries are the controls that replaced them.
SEARCH_SPACE = {
    "semantic_confidence": ("SEMANTIC_CONFIDENCE_THRESHOLD",
                            lambda r: float(r.uniform(0.20, 0.60))),
    # Relative to each blob's own field maximum, so this is a shape parameter,
    # not a calibration one: low keeps thin objects, high splits touching ones.
    "inner_distance_core_threshold": ("INNER_DISTANCE_CORE_THRESHOLD",
                                      lambda r: float(r.uniform(0.30, 0.70))),
    # Measured: 2-4% of real instances have a core near this floor, so it is
    # load-bearing rather than a formality.
    "minimum_core_area": ("MIN_INNER_DISTANCE_CORE_AREA",
                          lambda r: int(r.integers(2, 25))),
    "minimum_instance_area": ("MIN_INSTANCE_AREA",
                              lambda r: int(r.integers(8, 61))),
    "boundary_confidence": ("BOUNDARY_CONFIDENCE_THRESHOLD",
                            lambda r: float(r.uniform(0.30, 0.80))),
    # The floor on `inner_distance_instance_score`. Unlike the five above it
    # acts on ranking, not geometry: it removes decoded components whose field
    # does not agree with their own shape. On the synthetic scenes correct
    # objects score 0.90-0.98 and fragments/merges 0.53-0.86, so the useful
    # range is the band between those populations.
    "minimum_confidence": ("DEPLOYMENT_MIN_CONFIDENCE",
                           lambda r: float(r.uniform(0.00, 0.90))),
}


def score(m, cache, indices, iou):
    """Total TP/FP/FN over `indices` with the module's current thresholds."""
    tp = fp = fn = 0
    for i in indices:
        semantic, distance, boundary, target_instance, target_classes = cache[i]
        instance_map, classes, _, scores = m.decode_instances(
            semantic, distance, boundary
        )
        floor = float(getattr(m, "DEPLOYMENT_MIN_CONFIDENCE", 0.0) or 0.0)
        if floor > 0.0:
            instance_map, classes, scores, _ = m.filter_instances_by_confidence(
                instance_map, classes, semantic, scores, floor
            )
        counts = m.match_instances_at_iou(
            instance_map, classes, target_instance, target_classes, iou
        )
        for c in counts.values():
            tp += int(c.get("tp", 0))
            fp += int(c.get("fp", 0))
            fn += int(c.get("fn", 0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    m = load_model_module()
    import tensorflow as tf

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    m.REQUIRE_GPU = False

    output_dir = Path(os.environ["PCB_MODEL_OUTPUT_DIR"])
    model_path = output_dir / "best_model_v7_instance.keras"
    images_wanted = int(os.environ.get("PCB_FIT_IMAGES", "300"))
    draws = int(os.environ.get("PCB_FIT_DRAWS", "150"))
    iou = float(m.INSTANCE_EVALUATION_IOU)

    model = tf.keras.models.load_model(model_path, compile=False)
    arrays = m.load_split_arrays("val", array_dir=Path(os.environ["PCB_ARRAY_DIR"]))
    total = len(arrays["images"])
    count = min(images_wanted, total)
    print(f"model : {model_path.name}")
    print(f"images: {count} of {total} validation")
    print(f"draws : {draws}\n", flush=True)

    # ---- cache predictions once ------------------------------------------
    cache = {}
    started = time.time()
    for i in range(count):
        image = m.normalize_image(arrays["images"][i])[None, ...]
        outputs = m.unpack_model_outputs(model(image, training=False))
        semantic, distance, boundary = m.output_probabilities(outputs, 0)
        target_semantic, target_instance = m.sanitize_semantic_and_instances(
            np.asarray(arrays["semantic"][i], dtype=np.int32),
            np.asarray(arrays["instance"][i], dtype=np.int32),
        )
        cache[i] = (
            semantic.astype(np.float16), distance.astype(np.float16),
            boundary.astype(np.float16),
            target_instance,
            m.ground_truth_instance_classes(target_instance, target_semantic),
        )
        if (i + 1) % 50 == 0 or i + 1 == count:
            print(f"  cached {i + 1}/{count} "
                  f"({(i + 1) / (time.time() - started):.2f} img/s)", flush=True)

    # ---- halve the split -------------------------------------------------
    rng = np.random.default_rng(int(m.SEED))
    order = rng.permutation(count)
    fit_indices = sorted(order[: count // 2].tolist())
    holdout_indices = sorted(order[count // 2:].tolist())
    print(f"\nfit on {len(fit_indices)} images, verify on {len(holdout_indices)}\n")

    current = {name: getattr(m, attribute)
               for name, (attribute, _) in SEARCH_SPACE.items()}
    baseline_fit = score(m, cache, fit_indices, iou)
    baseline_holdout = score(m, cache, holdout_indices, iou)
    print(f"shipped thresholds  fit F1 {baseline_fit['f1']:.4f}   "
          f"holdout F1 {baseline_holdout['f1']:.4f}")
    print(f"  {current}\n", flush=True)

    # ---- random search on the fitting half -------------------------------
    best = (baseline_fit["f1"], dict(current))
    started = time.time()
    for draw in range(draws):
        candidate = {name: sampler(rng)
                     for name, (_, sampler) in SEARCH_SPACE.items()}
        for name, (attribute, _) in SEARCH_SPACE.items():
            setattr(m, attribute, candidate[name])
        result = score(m, cache, fit_indices, iou)
        if result["f1"] > best[0]:
            best = (result["f1"], dict(candidate))
            print(f"  draw {draw + 1:>3}: fit F1 {result['f1']:.4f}  "
                  f"P {result['precision']:.4f} R {result['recall']:.4f}  <- best",
                  flush=True)
        elif (draw + 1) % 25 == 0:
            print(f"  draw {draw + 1:>3}: best so far {best[0]:.4f} "
                  f"({(draw + 1) / (time.time() - started):.1f} draws/s)", flush=True)

    # ---- verify on the half that was never fitted ------------------------
    for name, (attribute, _) in SEARCH_SPACE.items():
        setattr(m, attribute, best[1][name])
    verified = score(m, cache, holdout_indices, iou)

    print("\n" + "=" * 74)
    print(f"{'':<22}{'fit F1':>10}{'holdout F1':>13}{'holdout P':>12}{'holdout R':>12}")
    print(f"{'shipped':<22}{baseline_fit['f1']:>10.4f}"
          f"{baseline_holdout['f1']:>13.4f}{baseline_holdout['precision']:>12.4f}"
          f"{baseline_holdout['recall']:>12.4f}")
    print(f"{'fitted':<22}{best[0]:>10.4f}"
          f"{verified['f1']:>13.4f}{verified['precision']:>12.4f}"
          f"{verified['recall']:>12.4f}")
    print("=" * 74)
    gain = verified["f1"] - baseline_holdout["f1"]
    print(f"held-out gain: {gain:+.4f} F1")
    if gain <= 0:
        print("  The winner did not survive the held-out half. That is the search\n"
              "  fitting noise, and the shipped thresholds should be kept.")
    print(f"\nbest thresholds: {json.dumps(best[1], indent=2)}")

    # A separate file, deliberately. results/threshold_fit.json is the V6.2 fit:
    # it names center_confidence and center_nms_radius, which V7's decoder does
    # not read, and its three surviving values were fitted against a different
    # grouping rule. Writing V7 results there would let a V6 fit and a V7 fit be
    # mistaken for one another.
    out = HERE / "results" / "threshold_fit_v7.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "images": count, "draws": draws,
        "shipped": {"values": current, "fit": baseline_fit,
                    "holdout": baseline_holdout},
        "fitted": {"values": best[1], "fit_f1": best[0], "holdout": verified},
        "held_out_gain_f1": gain,
        "caveat": "44 images are byte-identical between val and test; "
                  "thresholds fitted here contaminate ~10% of the test split.",
    }, indent=2), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
