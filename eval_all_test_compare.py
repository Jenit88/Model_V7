#!/usr/bin/env python3
"""Score the original and the fine-tuned V7 on both test splits together.

The overall comparison: the original 432-image test split and the new
664-image test split are concatenated into one 1,096-image test set
(~/data/pcb_v7_alltest_arrays, test arrays only; the sources are only read),
and both models go through the same m.evaluate("test") path as
eval_old_test_compare.py and eval_new_test_compare.py. Pooled mAP cannot be
averaged from the two separate results, so it is computed here directly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

HERE = Path(__file__).resolve().parent
OLD = Path("/home/u117134c/data/pcb_v62_arrays_repaired")
NEW = Path("/home/u117134c/data/pcb_v7_new3000_arrays")
ARRAY_DIR = Path("/home/u117134c/data/pcb_v7_alltest_arrays")
OUT = Path("/home/u117134c/Models/v7_all_test_eval_2026-10-12")
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


def build_arrays() -> None:
    if (ARRAY_DIR / "instance_dataset_metadata.json").is_file():
        return
    ARRAY_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ("X", "Y_semantic", "Y_instance"):
        parts = [np.load(root / f"{kind}_test.npy", mmap_mode="r") for root in (OLD, NEW)]
        assert len({p.dtype for p in parts}) == 1 and len({p.shape[1:] for p in parts}) == 1
        out = open_memmap(ARRAY_DIR / f"{kind}_test.npy", mode="w+", dtype=parts[0].dtype,
                          shape=(sum(len(p) for p in parts), *parts[0].shape[1:]))
        offset = 0
        for part in parts:
            out[offset:offset + len(part)] = part
            offset += len(part)
        out.flush()
        del out
    records = []
    for root in (OLD, NEW):
        for record in json.loads((root / "test_records.json").read_text()):
            records.append({**record, "index": len(records), "source": str(root)})
    (ARRAY_DIR / "test_records.json").write_text(json.dumps(records, indent=1))
    meta_old = json.loads((OLD / "instance_dataset_metadata.json").read_text())
    meta_new = json.loads((NEW / "instance_dataset_metadata.json").read_text())
    rows = [next(s for s in m["splits"] if s["split"] == "test") for m in (meta_old, meta_new)]
    meta = {k: v for k, v in meta_new.items() if k != "splits"}
    meta["array_dir"] = str(ARRAY_DIR)
    meta["dataset_root"] = "both test splits: original 432 + new 664"
    meta["splits"] = [{"split": "test", "images": sum(r["images"] for r in rows),
                       "instances": sum(r["instances"] for r in rows),
                       "parts": [{"source": str(src), "images": r["images"],
                                  "instances": r["instances"]}
                                 for src, r in zip((OLD, NEW), rows)]}]
    (ARRAY_DIR / "instance_dataset_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"combined test arrays: {len(records)} images -> {ARRAY_DIR}", flush=True)


def sha256_prefix(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main() -> None:
    build_arrays()
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
        print(f"\n=== {name} on all test ({path.name}, sha256 {sha256_prefix(path)})",
              flush=True)
        semantic, instance = m.evaluate("test")
        overall = instance.get("overall", {})
        results[name] = {
            "images": len(records),
            "mask_map50_95": instance.get("mask_map50_95"),
            "mask_map50": instance.get("mask_map50"),
            "mask_map75": instance.get("mask_map75"),
            **{k: overall.get(k) for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
            "minutes": round((time.time() - started) / 60, 1),
        }
        comparison.write_text(json.dumps(results, indent=2, default=float))
        print("RESULT", name, json.dumps(results[name], default=float), flush=True)
    print("\nALL DONE ->", comparison, flush=True)


if __name__ == "__main__":
    main()
