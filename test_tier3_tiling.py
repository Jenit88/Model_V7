#!/usr/bin/env python3
"""Checks for V7 tiled inference, including a real forward pass.

The properties that matter are coverage and seams. A gap in coverage silently
drops objects; a visible seam is read by the decoder as a boundary, so it
manufactures instance splits exactly where two tiles meet -- which would look
like an over-segmentation bug anywhere except its actual cause.

    ./test_tier3_tiling.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location(
        "model_v7", HERE / "model_v7.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_v7"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    m = load()
    m.REQUIRE_GPU = False
    checks = []

    # --- geometry ---------------------------------------------------------
    # The real case: a 1536x1024 board, the shape of every failure image.
    positions = m.tile_grid(1024, 1536, m.IMG_SIZE, m.TILE_OVERLAP)
    covered = np.zeros((1024, 1536), dtype=np.int32)
    for top, left in positions:
        covered[top:top + m.IMG_SIZE, left:left + m.IMG_SIZE] += 1
    checks.append((
        f"1536x1024 fully covered by {len(positions)} tiles",
        int(covered.min()) >= 1,
        int(covered.min()),
    ))
    checks.append((
        "no tile runs outside the frame",
        all(
            top + m.IMG_SIZE <= 1024 and left + m.IMG_SIZE <= 1536
            for top, left in positions
        ),
        len(positions),
    ))
    # Effective magnification versus letterboxing the whole board into 512.
    letterbox_scale = m.IMG_SIZE / 1536
    checks.append((
        f"objects arrive {1.0 / letterbox_scale:.1f}x larger than letterboxed",
        (1.0 / letterbox_scale) > 2.5,
        1.0 / letterbox_scale,
    ))

    # An image smaller than one tile must still produce exactly one tile.
    small = m.tile_grid(300, 400, m.IMG_SIZE, m.TILE_OVERLAP)
    checks.append(("undersized image yields one tile", small == [(0, 0)], len(small)))

    # --- blend weights ----------------------------------------------------
    weight = m._tile_blend_weight(m.IMG_SIZE, m.TILE_BLEND_MARGIN)
    checks.append((
        "blend weight is 1 at the tile centre",
        abs(float(weight[m.IMG_SIZE // 2, m.IMG_SIZE // 2]) - 1.0) < 1e-6,
        float(weight[m.IMG_SIZE // 2, m.IMG_SIZE // 2]),
    ))
    checks.append((
        "blend weight tapers to ~0 at the edge",
        float(weight[0, 0]) < 0.01,
        float(weight[0, 0]),
    ))

    # --- a real forward pass ----------------------------------------------
    # Small enough to build on CPU quickly, large enough to need several tiles.
    model = m.build_model_v7_instance()
    height, width = 700, 900
    rng = np.random.default_rng(0)
    image = rng.random((height, width, 3), dtype=np.float32)

    semantic, distance, boundary = m.predict_fields_tiled(
        model, image, batch_size=2
    )
    checks.append((
        f"stitched semantic is source-shaped {semantic.shape[:2]}",
        semantic.shape == (height, width, m.NUM_CLASSES),
        semantic.shape[0],
    ))
    checks.append((
        "stitched distance is [H,W,1] in [0,1]",
        distance.shape == (height, width, 1)
        and float(distance.min()) >= 0.0
        and float(distance.max()) <= 1.0,
        float(distance.max()),
    ))
    checks.append((
        "semantic probabilities still sum to one after blending",
        bool(np.allclose(semantic.sum(axis=-1), 1.0, atol=1e-4)),
        float(np.abs(semantic.sum(axis=-1) - 1.0).max()),
    ))
    checks.append((
        "no NaN anywhere in the stitched fields",
        bool(
            np.all(np.isfinite(semantic))
            and np.all(np.isfinite(distance))
            and np.all(np.isfinite(boundary))
        ),
        0.0,
    ))

    # Seam check. A seam is indistinguishable from an object boundary to the
    # decoder, so this is what keeps tiling from inventing instance splits
    # along tile edges.
    #
    # The comparison has to be against the field's own natural variation, not
    # an absolute threshold: on a textured input the field varies everywhere,
    # and an absolute limit would fail on ordinary content while passing a
    # genuine seam on smooth content. A constant image removes the content
    # variation, leaving tile geometry as the only thing that can produce a
    # step -- and then seam columns are compared against non-seam columns.
    flat = np.full((height, width, 3), 0.5, dtype=np.float32)
    _, flat_distance, _ = m.predict_fields_tiled(model, flat, batch_size=2)
    row = flat_distance[height // 2, :, 0]
    gradient = np.abs(np.diff(row))

    seam_columns = set()
    for _, left in m.tile_grid(height, width, m.IMG_SIZE, m.TILE_OVERLAP):
        for edge in (left, left + m.IMG_SIZE):
            for offset in range(-2, 3):
                if 0 <= edge + offset < len(gradient):
                    seam_columns.add(edge + offset)
    seam_index = np.array(sorted(seam_columns), dtype=int)
    other_index = np.setdiff1d(np.arange(len(gradient)), seam_index)

    seam_max = float(gradient[seam_index].max()) if seam_index.size else 0.0
    other_max = float(gradient[other_index].max()) if other_index.size else 0.0
    checks.append((
        f"seams no sharper than the field itself "
        f"({seam_max:.5f} vs {other_max:.5f})",
        seam_max <= max(other_max * 3.0, 0.02),
        seam_max - other_max,
    ))

    # The decoder must accept the stitched fields unchanged.
    instance_map, classes, confidence, scores = m.decode_instances(
        semantic, distance, boundary
    )
    checks.append((
        "decode_instances accepts stitched fields",
        instance_map.shape == (height, width),
        int(len(classes)),
    ))

    width_text = max(len(text) for text, _, _ in checks)
    failed = 0
    for text, ok, detail in checks:
        print(f"  {text:<{width_text}}  {'ok' if ok else 'FAIL'}   ({detail})")
        failed += 0 if ok else 1
    print()
    if failed:
        raise SystemExit(f"{failed} tiling check(s) failed")
    print("all tiling checks pass")


if __name__ == "__main__":
    main()
