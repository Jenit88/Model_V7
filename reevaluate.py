#!/usr/bin/env python3
"""Re-run the complete val and test evaluations against the current decoder.

Used to measure a decoder change on the finished checkpoint without retraining.
Writes into performance/<split>_<tag>/ so the original results stay intact and
the two can be compared directly.

    PCB_EVAL_TAG=drop_unassigned ./run.sh reevaluate
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

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

    output_dir = Path(os.environ["PCB_MODEL_OUTPUT_DIR"])
    m.MODEL_FOR_INFERENCE = output_dir / "best_model_v7_instance.keras"
    m.ARRAY_DIR = Path(os.environ["PCB_ARRAY_DIR"])
    m.EVALUATION_MAX_IMAGES = int(os.environ.get("PCB_EVAL_IMAGES", "0"))
    m.USE_TTA_AT_EVALUATION = (
        os.environ.get("PCB_USE_TTA", "0").strip().lower() in ("1", "true", "yes")
    )
    tag = os.environ.get("PCB_EVAL_TAG", "reeval")

    # Only applied when fit_thresholds.py's winner survived its held-out half;
    # improve.sh reads that back and sets this rather than trusting the search.
    applied = None
    if os.environ.get("PCB_APPLY_FITTED_THRESHOLDS", "0") == "1":
        fitted = Path("results/threshold_fit.json")
        if fitted.is_file():
            values = json.loads(fitted.read_text())["fitted"]["values"]
            attributes = {
                "semantic_confidence": "SEMANTIC_CONFIDENCE_THRESHOLD",
                "center_confidence": "CENTER_CONFIDENCE_THRESHOLD",
                "center_nms_radius": "CENTER_NMS_RADIUS",
                "minimum_instance_area": "MIN_INSTANCE_AREA",
                "boundary_confidence": "BOUNDARY_CONFIDENCE_THRESHOLD",
            }
            for key, attribute in attributes.items():
                if key in values:
                    setattr(m, attribute, values[key])
            applied = values

    print(f"model  : {m.MODEL_FOR_INFERENCE.name}")
    print(f"policy : UNASSIGNED_PIXEL_POLICY = {m.UNASSIGNED_PIXEL_POLICY!r}")
    print(f"TTA    : {m.USE_TTA_AT_EVALUATION}"
          f"{' (' + str(len(m.TTA_TRANSFORMS)) + ' transforms)' if m.USE_TTA_AT_EVALUATION else ''}")
    print(f"tag    : {tag}")
    print(f"thresh : {'fitted -> ' + json.dumps(applied) if applied else 'shipped values'}\n",
          flush=True)

    summary = {}
    for split in ("val", "test"):
        print(f"===== {split} =====", flush=True)
        semantic_report, instance_report = m.evaluate(split)
        overall = instance_report.get("overall", {})
        summary[split] = {
            "mask_map50_95": instance_report.get("mask_map50_95"),
            "mask_map50": instance_report.get("mask_map50"),
            "mask_map75": instance_report.get("mask_map75"),
            "f1": overall.get("f1"),
            "precision": overall.get("precision"),
            "recall": overall.get("recall"),
            "tp": overall.get("tp"),
            "fp": overall.get("fp"),
            "fn": overall.get("fn"),
            "per_class": {
                k: {kk: v[kk] for kk in ("precision", "recall", "f1", "tp", "fp", "fn")
                    if kk in v}
                for k, v in (instance_report.get("per_class") or {}).items()
            },
        }

    out = HERE / "results" / f"reevaluation_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"{'':<10}{'mAP50-95':>11}{'mAP50':>9}{'mAP75':>9}"
          f"{'F1':>9}{'prec':>9}{'recall':>9}")
    for split, s in summary.items():
        print(f"{split:<10}{s['mask_map50_95']:>11.4f}{s['mask_map50']:>9.4f}"
              f"{s['mask_map75']:>9.4f}{s['f1']:>9.4f}"
              f"{s['precision']:>9.4f}{s['recall']:>9.4f}")
    print("=" * 72)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
