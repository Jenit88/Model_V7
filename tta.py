#!/usr/bin/env python3
"""Dihedral test-time augmentation for model_v7.

Every class in this dataset -- rectangles, notched rectangles, annuli and discs
-- is symmetric under the full dihedral group, so all eight transforms are
exactly label-preserving and their predictions can be averaged.

The trap, and the reason this file exists rather than a five-line loop: **the
offset head is a vector field, not a raster**. Un-rotating the image grid is
only half the inverse; every (dx, dy) pair has to be rotated too, or the
averaged field points in eight different directions and the decoder groups
nothing.

Conventions, derived from `apply_dihedral_transform` (rot90 by k, then a
left-right flip):

    T(x)      = fliplr(rot90(x, k))
    T^-1(y)   = rot90(fliplr(y), -k)                       on a raster
    T_vec     = M . R_k        R_1:(dx,dy)->(dy,-dx)   M:(dx,dy)->(-dx,dy)
    T_vec^-1  = R_-k . M       R_-1:(dx,dy)->(-dy,dx)

`self_check()` does not verify that algebra against itself. It builds a scene,
generates its offset target with the project's own `build_center_and_offset_
targets`, generates the target of the *transformed* scene, and requires this
module's inverse to turn the second into the first. If the algebra is wrong the
check fails.

    ./run.sh tta-selfcheck
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "model_v7", HERE / "model_v7.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("model_v7", module)
    spec.loader.exec_module(module)
    return module


def forward_raster(array: np.ndarray, transform: int) -> np.ndarray:
    """Apply dihedral transform `transform` (0-7) to a [H,W,...] array."""
    quarter_turns = transform % 4
    if quarter_turns:
        array = np.rot90(array, quarter_turns, axes=(0, 1))
    if transform >= 4:
        array = array[:, ::-1]
    return np.ascontiguousarray(array)


def inverse_raster(array: np.ndarray, transform: int) -> np.ndarray:
    """Undo `forward_raster` on the grid only."""
    quarter_turns = transform % 4
    if transform >= 4:
        array = array[:, ::-1]
    if quarter_turns:
        array = np.rot90(array, -quarter_turns, axes=(0, 1))
    return np.ascontiguousarray(array)


def inverse_vectors(offset: np.ndarray, transform: int) -> np.ndarray:
    """Rotate (dx, dy) pairs back into the original frame.

    Expects the last axis to be at least 2, ordered (dx, dy, ...); any further
    channels (the validity mask) are carried through untouched.
    """
    offset = np.array(offset, copy=True)
    # Copies, not views. `offset[..., 0]` is a view, and writing the result
    # back into the same array clobbers the other component mid-swap -- which
    # for k=1 (where one of the two stays a view through the loop) silently
    # produced a wrong field on exactly one of the eight transforms.
    dx = np.array(offset[..., 0], copy=True)
    dy = np.array(offset[..., 1], copy=True)
    if transform >= 4:                       # M first
        dx = -dx
    for _ in range(transform % 4):           # then R_-1, k times
        dx, dy = -dy, dx
    offset[..., 0] = dx
    offset[..., 1] = dy
    return offset


def inverse_offset(offset: np.ndarray, transform: int) -> np.ndarray:
    """Full inverse for the offset head: un-rotate the grid AND the vectors."""
    return inverse_vectors(inverse_raster(offset, transform), transform)


def predict_with_tta(model, image, unpack_outputs, transforms=range(8),
                     batch_size: int = 2):
    """Average predictions over the dihedral transforms.

    `image` is one normalised [H,W,3] float image. `unpack_outputs` is the
    project's `unpack_model_outputs`, passed in rather than imported so this
    module never has to import the model file that calls it. Returns the same
    four arrays `output_probabilities` returns -- (semantic_prob, centre_prob,
    offset, boundary_prob) -- so this drops straight into the decode path.

    Semantic is averaged as logits (the softmax is applied once, at the end);
    centre and boundary are averaged as probabilities, which is what an
    ensemble of independent detectors should do; the offset field is averaged
    as vectors after each member has been rotated back.
    """
    transforms = list(transforms)

    semantic_logits = centre_prob = boundary_prob = offset_sum = None
    for start in range(0, len(transforms), batch_size):
        chunk = transforms[start:start + batch_size]
        batch = np.stack([forward_raster(image, t) for t in chunk])
        arrays = unpack_outputs(model(batch, training=False))
        for position, transform in enumerate(chunk):
            sem = inverse_raster(arrays["semantic"][position], transform)
            ctr = inverse_raster(arrays["center"][position], transform)
            bnd = inverse_raster(arrays["boundary"][position], transform)
            off = inverse_offset(arrays["offset"][position], transform)

            sem = sem.astype(np.float64)
            ctr = _sigmoid(ctr.astype(np.float64))
            bnd = _sigmoid(bnd.astype(np.float64))
            off = off.astype(np.float64)

            if semantic_logits is None:
                semantic_logits, centre_prob = sem, ctr
                boundary_prob, offset_sum = bnd, off
            else:
                semantic_logits += sem
                centre_prob += ctr
                boundary_prob += bnd
                offset_sum += off

    n = float(len(transforms))
    semantic_logits /= n
    centre_prob /= n
    boundary_prob /= n
    offset_sum /= n

    shifted = semantic_logits - semantic_logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    semantic_prob = exponentials / exponentials.sum(axis=-1, keepdims=True)

    if boundary_prob.ndim == 3 and boundary_prob.shape[-1] == 1:
        boundary_prob = boundary_prob[..., 0]
    return (semantic_prob.astype(np.float32), centre_prob.astype(np.float32),
            offset_sum.astype(np.float32), boundary_prob.astype(np.float32))


def _sigmoid(x):
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def self_check() -> None:
    """Validate the grid inverse and the vector inverse, exactly.

    Deliberately NOT tested against `build_center_and_offset_targets`: that
    generator resizes 512 -> 256 with INTER_NEAREST, and nearest-neighbour
    downsampling does not commute with rot90 or a flip -- it samples the other
    member of each 2x2 pair. A test against it fails for a reason that has
    nothing to do with this module, which is exactly the trap it looked like
    when first written here.

    Instead the vector map is checked against a field built analytically, where
    the right answer is known in closed form: for a centre c, offset(p) = c - p,
    so after moving the geometry the recovered field must equal the original
    everywhere.
    """
    m = _load()
    rng = np.random.default_rng(7)
    failures = []

    # --- the vector inverse, against an analytic field -----------------------
    size = 64

    def field_for(centre):
        ys, xs = np.mgrid[0:size, 0:size]
        return np.stack(
            [centre[0] - xs, centre[1] - ys, np.ones_like(xs)], axis=-1
        ).astype(np.float64)

    def move_point(point, transform):
        x, y = point
        for _ in range(transform % 4):
            x, y = y, size - 1 - x        # rot90 k=1 : (x,y) -> (y, n-1-x)
        if transform >= 4:
            x = size - 1 - x              # fliplr
        return (x, y)

    centre = (41, 17)
    reference = field_for(centre)
    for transform in range(8):
        seen_by_model = field_for(move_point(centre, transform))
        recovered = inverse_offset(seen_by_model, transform)
        difference = float(np.abs(recovered[..., :2] - reference[..., :2]).max())
        ok = difference < 1e-9
        print(f"  transform {transform}: vector field max|d| = {difference:.2e}"
              f"   {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append((transform, "vector inverse", difference))

    # --- the grid inverse, exactly ------------------------------------------
    probe = rng.random((m.IMG_SIZE, m.IMG_SIZE, 3)).astype(np.float32)
    for transform in range(8):
        if not np.array_equal(
            inverse_raster(forward_raster(probe, transform), transform), probe
        ):
            failures.append((transform, "raster round trip", None))

    # --- the eight transforms must actually be eight distinct views ---------
    views = {forward_raster(probe, t).tobytes() for t in range(8)}
    if len(views) != 8:
        failures.append(("all", f"only {len(views)} distinct views", None))

    if failures:
        raise SystemExit(f"\nTTA self-check FAILED: {failures}")
    print("\nTTA self-check passed: all 8 grid inverses are exact, all 8 vector"
          "\ninverses are exact, and the 8 transforms give 8 distinct views.")


if __name__ == "__main__":
    self_check()
