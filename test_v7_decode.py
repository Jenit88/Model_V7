#!/usr/bin/env python3
"""Contracts for the two V7 decode changes, so neither can silently revert.

Both were found by measurement rather than by reading, and both are the kind of
defect that leaves every existing test passing:

* the instance score was degenerate -- the target normalises every instance to
  peak at exactly 1.0, so "peak predicted distance" returned ~1.0 for a correct
  object, a merged pair and a rim sliver alike, and mask AP is computed from
  that ordering;
* core filtering and expansion were global rather than per foreground blob, so a
  small object beside a large one lost its core to a global area floor and had
  its pixels handed to the neighbour's core.

    ./test_v7_decode.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("model_v7", HERE / "model_v7.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_v7"] = module
    spec.loader.exec_module(module)
    return module


def box(canvas, x0, y0, x1, y1, value):
    canvas[y0:y1, x0:x1] = value


def disc(canvas, cx, cy, radius, value):
    ys, xs = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    canvas[(xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius] = value


def oracle_fields(m, instance_map):
    """The fields a perfect model would emit for this scene."""
    semantic = (instance_map > 0).astype(np.int32)
    semantic_target, distance, boundary = m.build_all_targets(semantic, instance_map)
    probabilities = np.zeros(
        (m.IMG_SIZE, m.IMG_SIZE, m.NUM_CLASSES), dtype=np.float32
    )
    probabilities[..., 0] = (semantic_target == 0).astype(np.float32)
    probabilities[..., 1] = (semantic_target > 0).astype(np.float32)
    boundary = np.asarray(boundary, dtype=np.float32)
    if boundary.ndim == 3:
        boundary = boundary[..., 0]
    return probabilities, distance, boundary


def score_of(m, mask, field, confidence):
    rows, columns = np.nonzero(mask)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    return m.inner_distance_instance_score(
        mask[y0:y1, x0:x1], field[y0:y1, x0:x1], confidence[y0:y1, x0:x1]
    )


def test_score_separates_correct_from_wrong(m):
    """A wrong grouping must score below every correct one, on a perfect field.

    This is the contract the replaced score failed: it returned 1.0000 for two
    of four correct objects and 0.8593 for the smallest, a spread that came from
    upsampling rather than from quality, and it had no way to rank a fragment
    below the object it was torn from.
    """
    size = m.IMG_SIZE
    instance_map = np.zeros((size, size), dtype=np.int32)
    box(instance_map, 40, 40, 170, 170, 1)          # 130 px
    disc(instance_map, 300, 90, 8, 2)               # 16 px across
    box(instance_map, 60, 300, 140, 400, 3)         # a touching pair
    box(instance_map, 140, 300, 220, 400, 4)

    _, distance, _ = oracle_fields(m, instance_map)
    field = m.resize_channels(distance[..., 0:1], size, size)[..., 0]
    confidence = np.ones((size, size), dtype=np.float32)

    correct = [
        score_of(m, instance_map == k, field, confidence) for k in (1, 2, 3, 4)
    ]

    rectangle = instance_map == 1
    half = rectangle.copy()
    half[:, 105:] = False
    quarter = rectangle.copy()
    quarter[105:, :] = False
    quarter[:, 105:] = False
    sliver = rectangle.copy()
    sliver[46:164, 46:164] = False
    merged = (instance_map == 3) | (instance_map == 4)
    wrong = [
        score_of(m, mask, field, confidence)
        for mask in (half, quarter, sliver, merged)
    ]

    assert min(correct) > max(wrong), (
        "instance score does not separate correct groupings from wrong ones: "
        f"worst correct {min(correct):.4f}, best wrong {max(wrong):.4f}"
    )
    # Guard the property that actually broke: a score that cannot vary cannot
    # rank, and mask AP is a ranked metric.
    assert max(correct) - min(wrong) > 0.20, (
        "instance score barely varies across correct and wrong groupings: "
        f"range {max(correct) - min(wrong):.4f}"
    )
    print(
        "  score separates correct from wrong "
        f"(worst correct {min(correct):.3f} > best wrong {max(wrong):.3f})"
    )


def test_small_object_beside_large_one_survives(m):
    """A blob whose core is below the area floor keeps its own pixels.

    The global filter dropped every sub-floor core whenever any core anywhere
    cleared the floor, and global expansion then handed the orphaned blob to the
    nearest *other* core. Measured on the test split, 2-4% of real instances
    have a sub-floor core at head resolution, so this was roughly two objects
    per image, not a corner case.
    """
    size = m.IMG_SIZE
    instance_map = np.zeros((size, size), dtype=np.int32)
    box(instance_map, 40, 40, 200, 200, 1)          # large, ample core
    disc(instance_map, 215, 120, 3, 2)              # tiny, sub-floor core

    _, distance, _ = oracle_fields(m, instance_map)
    field = m.resize_channels(distance[..., 0:1], size, size)[..., 0]
    foreground = instance_map > 0

    grouped = m.watershed_instances(foreground, field)
    found = [int(v) for v in np.unique(grouped) if v > 0]
    assert len(found) == 2, (
        f"expected the small object to survive as its own instance; got "
        f"{len(found)} instance(s)"
    )

    small_truth = instance_map == 2
    overlaps = [
        int((grouped == v)[small_truth].sum()) / int(small_truth.sum())
        for v in found
    ]
    assert max(overlaps) > 0.5, (
        "the small object was absorbed rather than kept as its own instance"
    )
    print("  a sub-floor core beside a large object is kept, not absorbed")


def test_instances_are_connected(m):
    """No instance may span two separated blobs.

    Expansion is Euclidean and the old 51 px cap is gone, so without a
    connectivity constraint a core could claim pixels across a gap and produce a
    mask made of two disjoint pieces.
    """
    size = m.IMG_SIZE
    instance_map = np.zeros((size, size), dtype=np.int32)
    box(instance_map, 30, 30, 170, 170, 1)
    disc(instance_map, 260, 100, 4, 2)
    disc(instance_map, 300, 260, 5, 3)
    box(instance_map, 360, 360, 470, 470, 4)

    _, distance, _ = oracle_fields(m, instance_map)
    field = m.resize_channels(distance[..., 0:1], size, size)[..., 0]
    grouped = m.watershed_instances(instance_map > 0, field)

    for value in [int(v) for v in np.unique(grouped) if v > 0]:
        count, _ = cv2.connectedComponents(
            (grouped == value).astype(np.uint8), connectivity=8
        )
        assert count == 2, (
            f"instance {value} is made of {count - 1} disconnected pieces"
        )
    print("  every decoded instance is a single connected component")


def test_loss_weight_is_calibrated(m):
    """The inner-distance head must hold a real share of the loss budget.

    It replaces centre+offset, which held 29.1% of V6.2's measured budget. At
    the inherited weight of 1.00 it would hold 3.8%, because a Huber on a [0,1]
    quantity is an order of magnitude smaller than the losses beside it.
    """
    semantic_contribution = 0.0751          # measured, Tier-1 epoch 73
    boundary_contribution = 0.30 * 0.1694
    mid_training_raw = 0.0050               # measured on real targets
    contribution = float(m.INNER_DISTANCE_LOSS_WEIGHT) * mid_training_raw
    share = contribution / (
        semantic_contribution + boundary_contribution + contribution
    )
    assert 0.18 <= share <= 0.36, (
        "INNER_DISTANCE_LOSS_WEIGHT leaves the only head that separates "
        f"instances at {share:.1%} of the loss budget; centre+offset held 29.1%"
    )
    print(f"  inner-distance head holds {share:.1%} of the loss budget")


def test_mosaic_is_label_safe(m):
    """A mosaic must not collide instance ids or leave sub-floor phantom labels.

    Two objects from different source boards carry the same id in their own
    frames. Tiling without re-basing would merge them into one instance spanning
    two quadrants. Halving a diameter also quarters an area, so an object can
    fall under MIN_AUGMENTED_INSTANCE_AREA -- and a label too small to be a valid
    target must leave the semantic map with it, not linger as a phantom.
    """
    size = m.IMG_SIZE
    samples = []
    for k in range(4):
        instance_map = np.zeros((size, size), dtype=np.int32)
        box(instance_map, 40, 40, 200, 200, 1)      # same ids in every quadrant
        disc(instance_map, 300, 300, 30, 2)
        disc(instance_map, 400, 120, 2, 3)          # sub-floor once halved
        semantic = (instance_map > 0).astype(np.int32)
        image = np.zeros((size, size, 3), dtype=np.float32)
        samples.append((image, semantic, instance_map))

    image, semantic, instance = m.mosaic_samples(samples)
    assert image.shape == (size, size, 3)
    assert instance.shape == (size, size)

    ids = [int(v) for v in np.unique(instance) if v > 0]
    assert len(ids) >= 8, (
        f"expected at least two surviving objects per quadrant; got {len(ids)}"
    )
    for value in ids:
        count, _ = cv2.connectedComponents(
            (instance == value).astype(np.uint8), connectivity=8
        )
        assert count == 2, (
            f"instance {value} spans more than one quadrant -- ids were not "
            "re-based per quadrant"
        )
    orphaned = int(((semantic > 0) & (instance == 0)).sum())
    assert orphaned == 0, (
        f"{orphaned} semantic pixels have no instance id: a dropped object was "
        "left in the semantic map"
    )
    print(f"  mosaic re-bases ids and drops sub-floor objects ({len(ids)} kept)")


def main() -> None:
    m = load()
    print("V7 decode contracts")
    test_score_separates_correct_from_wrong(m)
    test_small_object_beside_large_one_survives(m)
    test_instances_are_connected(m)
    test_loss_weight_is_calibrated(m)
    test_mosaic_is_label_safe(m)
    print("all V7 decode contracts hold")


if __name__ == "__main__":
    main()
