#!/usr/bin/env python3
"""Publish the metrics Keras' TensorBoard callback is not writing.

The live run writes only `batch_*` scalars: its `validation/` event file is 78
bytes -- a header and nothing else -- so there are no epoch curves, no validation
curves, and none of the instance metrics that actually say how the model is
doing (mAP50-95, mAP50, precision, recall, F1, per-class).

Those numbers are all on disk already, in `training_log.csv` (every epoch) and
`instance_checkpoint_history.json` (every instance evaluation). This reads both
and writes its own event files into `<output>/tensorboard_metrics`, which
TensorBoard serves alongside the run's own directory.

**It never imports or touches model_v7.py.** That matters: the training process
re-imports that module on every watchdog restart, so editing it mid-run means a
crash resumes into half-edited code. This is a reader, not a patch.

    ./tb_export.py                      # one pass over whatever exists now
    ./tb_export.py --watch              # re-export every 60s while training
    ./tb_export.py --output-dir DIR     # default: $PCB_MODEL_OUTPUT_DIR

Then point TensorBoard at the run directory; it picks up both subdirectories:

    tensorboard --logdir ~/Models/Model_v7_scratch_rtx --bind_all --port 6006
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import tensorflow as tf

# Written under these groups so the TensorBoard sidebar sorts them usefully
# rather than as one flat alphabetical list.
GROUPS = (
    ("loss", ("loss", "semantic_loss", "inner_distance_loss", "boundary_loss")),
    ("semantic", ("semantic_mean_iou", "semantic_foreground_miou",
                  "semantic_pixel_accuracy")),
    ("boundary", ("boundary_boundary_f1",)),
)


def scalar_rows(csv_path: Path):
    """Every epoch row from training_log.csv, tolerating a resumed header."""
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            raw = (row.get("epoch") or "").strip()
            if not raw or raw == "epoch":
                continue
            try:
                row["epoch"] = int(float(raw))
            except ValueError:
                continue
            rows.append(row)
    return rows


def as_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # A run logs `nan` on epochs with no instance evaluation; writing those
    # would put gaps in the curve as NaN points rather than simply no point.
    return None if number != number else number


def export(output_dir: Path) -> tuple[int, int]:
    log_dir = output_dir / "tensorboard_metrics"
    rows = scalar_rows(output_dir / "training_log.csv")

    history_path = output_dir / "instance_checkpoint_history.json"
    history = []
    if history_path.is_file() and history_path.stat().st_size:
        try:
            history = json.loads(history_path.read_text())
        except json.JSONDecodeError:
            history = []          # being rewritten right now; next pass gets it

    # Every pass writes a complete new set of event files, so the previous set
    # is deleted once this one is written. Nothing deleted them before: by
    # 2026-10-05, 5,156 passes had left 15,468 files, and TensorBoard's data
    # server, which keeps every file open, hit its 10,240-file limit and showed
    # no data at all. Listed before the writers exist, so only old sets match.
    previous_files = [
        path
        for name in ("train", "validation", "instance")
        for path in (log_dir / name).glob("events.out.tfevents.*")
    ]

    train_writer = tf.summary.create_file_writer(str(log_dir / "train"))
    val_writer = tf.summary.create_file_writer(str(log_dir / "validation"))
    instance_writer = tf.summary.create_file_writer(str(log_dir / "instance"))

    grouped = {name: group for group, names in GROUPS for name in names}

    with train_writer.as_default():
        for row in rows:
            step = row["epoch"]
            for key, value in row.items():
                if key == "epoch" or key.startswith("val_"):
                    continue
                number = as_float(value)
                if number is None:
                    continue
                group = grouped.get(key, "other")
                tf.summary.scalar(f"{group}/{key}", number, step=step)
    train_writer.flush()

    with val_writer.as_default():
        for row in rows:
            step = row["epoch"]
            for key, value in row.items():
                if not key.startswith("val_"):
                    continue
                number = as_float(value)
                if number is None:
                    continue
                bare = key[4:]
                group = grouped.get(bare, "instance" if "instance" in bare else "other")
                tf.summary.scalar(f"{group}/{bare}", number, step=step)
    val_writer.flush()

    # The instance evaluation is the one that answers "how is the model doing".
    # It runs every few epochs, so it gets its own writer and its own curves,
    # including the per-class breakdown the CSV does not carry at all.
    with instance_writer.as_default():
        for entry in history:
            step = int(entry.get("epoch", 0))
            for key in ("mask_map50_95", "mask_map50", "precision", "recall",
                        "f1", "semantic_foreground_miou", "selection_score"):
                number = as_float(entry.get(key))
                if number is not None:
                    tf.summary.scalar(f"instance/{key}", number, step=step)
            for key in ("tp", "fp", "fn"):
                number = as_float(entry.get(key))
                if number is not None:
                    tf.summary.scalar(f"counts/{key}", number, step=step)
            for class_name, stats in (entry.get("per_class") or {}).items():
                if not isinstance(stats, dict):
                    continue
                for key in ("precision", "recall", "f1"):
                    number = as_float(stats.get(key))
                    if number is not None:
                        tf.summary.scalar(
                            f"per_class_{key}/{class_name}", number, step=step
                        )
    instance_writer.flush()

    for writer in (train_writer, val_writer, instance_writer):
        writer.close()
    for path in previous_files:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return len(rows), len(history)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "PCB_MODEL_OUTPUT_DIR",
            str(Path.home() / "Models" / "Model_v7_scratch_rtx"),
        ),
    )
    parser.add_argument("--watch", action="store_true",
                        help="re-export every --interval seconds")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_dir():
        raise SystemExit(f"{output_dir} does not exist")

    while True:
        epochs, evaluations = export(output_dir)
        print(
            f"{time.strftime('%H:%M:%S')}  exported {epochs} epoch row(s), "
            f"{evaluations} instance evaluation(s) -> "
            f"{output_dir / 'tensorboard_metrics'}",
            flush=True,
        )
        if not args.watch:
            return
        time.sleep(max(10, int(args.interval)))


if __name__ == "__main__":
    main()
