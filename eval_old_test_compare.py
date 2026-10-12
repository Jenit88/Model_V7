#!/usr/bin/env python3
"""Score the original and the fine-tuned V7 on the original 432-image test split.

The companion of eval_new_test_compare.py: both models go through the same
m.evaluate("test") path, on the same arrays (pcb_v62_arrays_repaired, the test
split the original was trained and reported against), in one session. The
fine-tune never saw these photos: its training data was the original train
split plus the 3000_new_images train split.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARRAY_DIR = Path("/home/u117134c/data/pcb_v62_arrays_repaired")
OUT = Path("/home/u117134c/Models/v7_old_test_eval_2026-10-12")
os.environ["PCB_DATASET_ROOT"] = (
    "/mnt/c/Users/u117134/Desktop/Data_preprocessing/1000_images/Split_Data"
)
os.environ["PCB_ARRAY_DIR"] = str(ARRAY_DIR)
os.environ["PCB_MODEL_OUTPUT_DIR"] = str(OUT)
sys.path.insert(0, str(HERE))

MODELS = {
    "original_epoch105": Path(
        "/home/u117134c/Models/Model_v7_scratch_rtx/best_model_v7_instance.keras"
    ),
    "finetuned_epoch54": Path(
        "/home/u117134c/Models/Model_v7_finetune_new3000/"
        "best_fine_tuned_model_v7_instance.keras"
    ),
}


def sha256_prefix(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((ARRAY_DIR / "test_records.json").read_text())

    spec = importlib.util.spec_from_file_location("model_v7", HERE / "model_v7.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["model_v7"] = m
    spec.loader.exec_module(m)
    m.EVALUATION_MAX_IMAGES = 0

    comparison = OUT / "comparison.json"
    results = json.loads(comparison.read_text()) if comparison.is_file() else {}
    for name, path in MODELS.items():
        started = time.time()
        m.MODEL_FOR_INFERENCE = path
        m.MODEL_OUTPUT_DIR = OUT / name
        m.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name} on old test ({path.name}, sha256 {sha256_prefix(path)})",
              flush=True)
        semantic, instance = m.evaluate("test")
        overall = instance.get("overall", {})
        result = {
            "images": len(records),
            "mask_map50_95": instance.get("mask_map50_95"),
            "mask_map50": instance.get("mask_map50"),
            "mask_map75": instance.get("mask_map75"),
            **{k: overall.get(k) for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
            "per_class_ap": instance.get("per_class_ap") or instance.get("per_class"),
            "semantic_foreground_miou": semantic.get("foreground_miou"),
            "minutes": round((time.time() - started) / 60, 1),
        }
        results[name] = result
        comparison.write_text(json.dumps(results, indent=2, default=float))
        print("RESULT", name, json.dumps(
            {k: result[k] for k in ("mask_map50_95", "mask_map50", "mask_map75",
                                    "precision", "recall", "f1", "minutes")},
            default=float), flush=True)
    print("\nALL DONE ->", comparison, flush=True)


if __name__ == "__main__":
    main()
