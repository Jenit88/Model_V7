#!/usr/bin/env python3
"""Dihedral test-time augmentation for the V7 three-head model.

A separate module from `tta.py` rather than an edit to it: that file is on the
live V7 evaluation path, including the run training right now, and the two
models return different heads.

**The vector-field problem is gone.** `tta.py` needs `inverse_vectors` because
the offset head emits a vector field: un-rotating the raster is only half the
inverse, and every (dx, dy) pair has to be rotated too, or the averaged field
points eight ways and the decoder groups nothing. Its self-check caught a real
aliasing bug in exactly that step. The inner-distance field is a scalar per
pixel, so un-rotating the raster *is* the whole inverse and the entire class of
error disappears -- one of the quieter benefits of replacing the field.

Averaging follows the same rules: semantic in logit space with one softmax at
the end, boundary in probability space, and the distance field directly, since
it is already a bounded normalised quantity rather than a logit.
"""
from __future__ import annotations

import numpy as np

from tta import forward_raster, inverse_raster, _sigmoid  # noqa: F401


def predict_with_tta(model, image, unpack_outputs, transforms=range(8),
                     batch_size: int = 2):
    """Average predictions over the dihedral transforms.

    Returns the three arrays `output_probabilities` returns for V7 --
    (semantic_prob, inner_distance, boundary_prob) -- so it drops straight into
    the decode path.
    """
    transforms = list(transforms)

    semantic_logits = distance_sum = boundary_prob = None
    for start in range(0, len(transforms), batch_size):
        chunk = transforms[start:start + batch_size]
        batch = np.stack([forward_raster(image, t) for t in chunk])
        arrays = unpack_outputs(model(batch, training=False))
        for position, transform in enumerate(chunk):
            sem = inverse_raster(arrays["semantic"][position], transform)
            bnd = inverse_raster(arrays["boundary"][position], transform)
            # Scalar field: un-rotating the raster is the complete inverse.
            dist = inverse_raster(arrays["inner_distance"][position], transform)

            sem = sem.astype(np.float64)
            bnd = _sigmoid(bnd.astype(np.float64))
            # Already a sigmoid output from the head, so no activation here.
            dist = dist.astype(np.float64)

            if semantic_logits is None:
                semantic_logits, boundary_prob, distance_sum = sem, bnd, dist
            else:
                semantic_logits += sem
                boundary_prob += bnd
                distance_sum += dist

    n = float(len(transforms))
    semantic_logits /= n
    boundary_prob /= n
    distance_sum /= n

    shifted = semantic_logits - semantic_logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    semantic_prob = exponentials / exponentials.sum(axis=-1, keepdims=True)

    if boundary_prob.ndim == 3 and boundary_prob.shape[-1] == 1:
        boundary_prob = boundary_prob[..., 0]

    return (
        np.ascontiguousarray(semantic_prob.astype(np.float32)),
        np.ascontiguousarray(np.clip(distance_sum, 0.0, 1.0).astype(np.float32)),
        np.ascontiguousarray(boundary_prob.astype(np.float32)),
    )
