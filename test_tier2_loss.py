#!/usr/bin/env python3
"""Contract checks for MaskedInnerDistanceLoss, before it costs a retrain.

A regression loss that is quietly wrong does not announce itself: training
converges, the number falls, and the field it produces is useless. These assert
the properties the decoder actually depends on.

    ./test_tier2_loss.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def field(distance, mask):
    return np.stack([distance, mask], axis=-1)[None].astype(np.float32)


def main() -> None:
    m = load("model_v7")
    import tensorflow as tf

    loss = m.MaskedInnerDistanceLoss()
    size = 16
    checks = []

    # A plausible object: distance rising from rim to centre.
    ys, xs = np.mgrid[0:size, 0:size]
    radial = 1.0 - (np.hypot(xs - 7.5, ys - 7.5) / 7.5)
    target = np.clip(radial, 0.0, 1.0).astype(np.float32)
    mask = (target > 0.0).astype(np.float32)

    # 1. Perfect prediction costs nothing.
    value = float(loss(field(target, mask), target[None, ..., None]))
    checks.append(("perfect prediction is zero", abs(value) < 1e-6, value))

    # 2. Background is ignored entirely: changing predictions outside the mask
    #    must not move the loss, or the head would be trained on pixels that
    #    have no target.
    predicted = target.copy()
    predicted[mask == 0.0] = 0.9
    value_bg = float(loss(field(target, mask), predicted[None, ..., None]))
    checks.append(
        ("background predictions ignored", abs(value_bg) < 1e-6, value_bg)
    )

    # 3. Rim error costs more than the same error in the interior. This is the
    #    property that keeps touching objects separable -- the interior is the
    #    easy majority and would otherwise dominate.
    rim = (target > 0.0) & (target < 0.2)
    interior = target > 0.8
    rim_wrong = target.copy()
    rim_wrong[rim] += 0.3
    interior_wrong = target.copy()
    interior_wrong[interior] -= 0.3
    # Same number of perturbed pixels, so the comparison is like for like.
    count = min(int(rim.sum()), int(interior.sum()))
    rim_wrong = target.copy()
    rim_wrong[np.argwhere(rim)[:count, 0], np.argwhere(rim)[:count, 1]] += 0.3
    interior_wrong = target.copy()
    interior_wrong[
        np.argwhere(interior)[:count, 0], np.argwhere(interior)[:count, 1]
    ] -= 0.3
    rim_loss = float(loss(field(target, mask), rim_wrong[None, ..., None]))
    interior_loss = float(
        loss(field(target, mask), interior_wrong[None, ..., None])
    )
    checks.append((
        f"rim error costs more than interior ({rim_loss:.5f} > {interior_loss:.5f})",
        rim_loss > interior_loss,
        rim_loss - interior_loss,
    ))

    # 4. An empty image is zero, not NaN. Whole letterbox-bar-only crops occur.
    empty = np.zeros((size, size), dtype=np.float32)
    value_empty = float(loss(field(empty, empty), empty[None, ..., None]))
    checks.append((
        "empty foreground is zero, not NaN",
        np.isfinite(value_empty) and abs(value_empty) < 1e-6,
        value_empty,
    ))

    # 5. Scale invariance -- the property the offset field never had. The same
    #    shape at two sizes, predicted equally badly, must cost the same.
    small = np.clip(1.0 - np.hypot(xs - 7.5, ys - 7.5) / 3.0, 0.0, 1.0)
    small = small.astype(np.float32)
    small_mask = (small > 0.0).astype(np.float32)
    big_loss = float(
        loss(field(target, mask), np.clip(target - 0.2, 0, 1)[None, ..., None])
    )
    small_loss = float(
        loss(field(small, small_mask), np.clip(small - 0.2, 0, 1)[None, ..., None])
    )
    checks.append((
        f"same error costs the same at 2x scale "
        f"({big_loss:.5f} vs {small_loss:.5f})",
        abs(big_loss - small_loss) < 0.02,
        big_loss - small_loss,
    ))

    # 6. Gradients reach the prediction.
    variable = tf.Variable(np.zeros((1, size, size, 1), dtype=np.float32))
    with tf.GradientTape() as tape:
        current = loss(field(target, mask), variable)
    gradient = tape.gradient(current, variable)
    magnitude = float(tf.reduce_max(tf.abs(gradient)))
    checks.append((
        "gradient flows to the prediction", magnitude > 1e-6, magnitude
    ))

    width = max(len(text) for text, _, _ in checks)
    failed = 0
    for text, ok, detail in checks:
        print(f"  {text:<{width}}  {'ok' if ok else 'FAIL'}   ({detail:+.6g})")
        failed += 0 if ok else 1
    print()
    if failed:
        raise SystemExit(f"{failed} contract(s) failed")
    print("all inner-distance loss contracts hold")


if __name__ == "__main__":
    main()
