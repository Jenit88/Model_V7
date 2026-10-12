#!/usr/bin/env python3
"""Score the original and the fine-tuned V7 on the 3000_new_images test split.

Both models go through the same m.evaluate("test") path that produced the
original's 52.4% on 2026-09-29 (prepare_eval_new3000.py), on the same arrays,
in one session -- so the comparison is like for like even if model_v7.py's
evaluation code changed since then.

Two scopes per model:
  all_664  the complete test split
  no_dup_652  without the 12 test photos that also appear in the new train
              split, re-found with verify_combined_data.py's method (16x16
              board thumbnails, correlation >= 0.99, full-res mean |diff| < 3)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ARRAY_DIR = Path("/home/u117134c/data/pcb_v7_new3000_arrays")
OUT = Path("/home/u117134c/Models/v7_test_eval_2026-10-06")
os.environ["PCB_DATASET_ROOT"] = (
    "/mnt/c/Users/u117134/Desktop/Data_preprocessing/3000_new_images/split_data"
)
os.environ["PCB_ARRAY_DIR"] = str(ARRAY_DIR)
os.environ["PCB_MODEL_OUTPUT_DIR"] = str(OUT)
sys.path.insert(0, str(HERE))

MODELS = {
    "finetuned_epoch54": Path(
        "/home/u117134c/Models/Model_v7_finetune_new3000/"
        "best_fine_tuned_model_v7_instance.keras"
    ),
    "original_epoch105": Path(
        "/home/u117134c/Models/Model_v7_scratch_rtx/best_model_v7_instance.keras"
    ),
    # Best by the semantic monitor rather than by instance mAP; ranked 2nd on
    # the combined val. (best_fine_tuned_instance_model_v7.keras is
    # byte-identical to finetuned_epoch54, so it is not listed twice.)
    "finetuned_semantic_best": Path(
        "/home/u117134c/Models/Model_v7_finetune_new3000/"
        "best_fine_tuned_semantic_model_v7.keras"
    ),
}
DEFAULT_MODELS = ("finetuned_epoch54", "original_epoch105")
KNOWN_DUP_ROWS = {37, 118, 148, 197, 213, 379}   # the 6 the audit printed
FILL = 114


def board(x: np.ndarray) -> np.ndarray:
    keep = np.any(x != FILL, axis=2)
    rows, cols = np.flatnonzero(keep.any(1)), np.flatnonzero(keep.any(0))
    if rows.size == 0:
        return x
    return x[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def thumb(x: np.ndarray, size: int = 16) -> np.ndarray:
    g = cv2.cvtColor(np.ascontiguousarray(board(x)), cv2.COLOR_RGB2GRAY)
    t = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (t - t.mean()) / (t.std() + 1e-6)


def find_test_duplicates() -> list[dict]:
    test = np.load(ARRAY_DIR / "X_test.npy", mmap_mode="r")
    train = np.load(ARRAY_DIR / "X_train.npy", mmap_mode="r")
    a = np.stack([thumb(test[i]) for i in range(len(test))])
    b = np.stack([thumb(train[j]) for j in range(len(train))])
    hits = []
    c = a @ b.T / a.shape[1]
    for i, j in zip(*np.nonzero(c >= 0.99)):
        xa, xb = board(test[i]), board(train[j])
        if xa.shape != xb.shape:
            continue
        mad = float(np.abs(xa.astype(np.int16) - xb.astype(np.int16)).mean())
        if mad < 3.0:
            hits.append({"test_row": int(i), "train_row": int(j),
                         "corr16": round(float(c[i, j]), 4), "mean_abs_diff": round(mad, 3)})
    return hits


def sha256_prefix(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help=f"comma-separated, from: {', '.join(MODELS)}")
    chosen = [name.strip() for name in parser.parse_args().models.split(",") if name.strip()]
    unknown = [name for name in chosen if name not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}")

    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((ARRAY_DIR / "test_records.json").read_text())

    print("Finding test photos duplicated in the new train split...", flush=True)
    hits = find_test_duplicates()
    dup_rows = sorted({h["test_row"] for h in hits})
    for h in hits:
        h["test_image"] = Path(records[h["test_row"]]["image"]).name
    (OUT / "test_duplicates_in_new_train.json").write_text(json.dumps(hits, indent=2))
    print(f"  {len(dup_rows)} test rows: {dup_rows}", flush=True)
    dup_ok = len(dup_rows) == 12 and KNOWN_DUP_ROWS <= set(dup_rows)
    if not dup_ok:
        print("  WARNING: does not match the 2026-09-29 audit (12 rows incl. "
              f"{sorted(KNOWN_DUP_ROWS)}); skipping the 652 scope.", flush=True)
    keep_rows = np.array([r for r in range(len(records)) if r not in set(dup_rows)],
                         dtype=np.int64)

    spec = importlib.util.spec_from_file_location("model_v7", HERE / "model_v7.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["model_v7"] = m
    spec.loader.exec_module(m)
    m.EVALUATION_MAX_IMAGES = 0
    full_selection = m.selected_indices

    scopes = [("all_664", None)] + ([("no_dup_652", keep_rows)] if dup_ok else [])
    # A later run for other models adds to the same comparison.json.
    comparison = OUT / "comparison.json"
    results = json.loads(comparison.read_text()) if comparison.is_file() else {}
    for scope, rows in scopes:
        for name in chosen:
            path = MODELS[name]
            started = time.time()
            m.MODEL_FOR_INFERENCE = path
            m.MODEL_OUTPUT_DIR = OUT / name / scope
            m.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            m.selected_indices = (
                full_selection if rows is None else (lambda total, maximum, r=rows: r)
            )
            print(f"\n=== {name} on {scope} ({path.name}, sha256 {sha256_prefix(path)})",
                  flush=True)
            semantic, instance = m.evaluate("test")
            overall = instance.get("overall", {})
            result = {
                "images": len(records) if rows is None else int(len(rows)),
                "mask_map50_95": instance.get("mask_map50_95"),
                "mask_map50": instance.get("mask_map50"),
                "mask_map75": instance.get("mask_map75"),
                **{k: overall.get(k) for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
                "per_class_ap": instance.get("per_class_ap") or instance.get("per_class"),
                "semantic_foreground_miou": semantic.get("foreground_miou"),
                "minutes": round((time.time() - started) / 60, 1),
            }
            results.setdefault(scope, {})[name] = result
            comparison.write_text(json.dumps(results, indent=2, default=float))
            print("RESULT", scope, name, json.dumps(
                {k: result[k] for k in ("mask_map50_95", "mask_map50", "mask_map75",
                                        "precision", "recall", "f1", "minutes")},
                default=float), flush=True)
    m.selected_indices = full_selection
    print("\nALL DONE ->", OUT / "comparison.json", flush=True)


if __name__ == "__main__":
    main()
