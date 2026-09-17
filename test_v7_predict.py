#!/usr/bin/env python3
"""The public prediction path must run end to end on a real, untrained model.

Every other suite here tests a function in isolation. This one builds the model,
runs `predict_one_image`, and checks the artifacts land -- because the defect it
exists to catch was invisible to unit tests: `predict_one_image` referenced
`center_probabilities_small`, a name deleted with the centre head, so every real
prediction raised NameError while all component tests passed.

Random weights are the point, not a limitation: this asserts the *interface*,
not accuracy. No checkpoint is needed, and it runs on CPU.

    ./test_v7_predict.py
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("model_v7", HERE / "model_v7.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_v7"] = module
    spec.loader.exec_module(module)
    return module


def synthetic_board(width, height, seed=0):
    """A board-like image at a non-square, non-512 size, so letterboxing runs."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    image[..., 1] = 90
    for _ in range(24):
        x = int(rng.integers(0, max(1, width - 40)))
        y = int(rng.integers(0, max(1, height - 40)))
        w = int(rng.integers(10, 34))
        h = int(rng.integers(10, 34))
        image[y:y + h, x:x + w] = rng.integers(150, 255, size=3, dtype=np.uint8)
    return image


def test_predict_one_image_runs(m, model):
    """The whole public path: letterbox, predict, decode, restore, write."""
    image = synthetic_board(900, 620)
    with tempfile.TemporaryDirectory() as directory:
        # predict_one_image reads from disk, which is part of what this checks:
        # the board is deliberately non-square and not 512, so letterboxing and
        # the restore back to native geometry both run for real.
        image_path = Path(directory) / "board.png"
        import cv2
        cv2.imwrite(str(image_path), image[..., ::-1])
        output_dir = Path(directory) / "prediction"
        result = m.predict_one_image(model, image_path, output_dir)

        assert isinstance(result, dict), "predict_one_image must return a summary"
        produced = sorted(p.name for p in output_dir.iterdir())
        assert produced, "predict_one_image wrote no artifacts"

        # The centre heatmap is gone; V7's per-pixel instance evidence is the
        # inner-distance field, and the artifact must follow the model.
        assert not any("centre" in n or "center" in n for n in produced), (
            f"a centre artifact survives in V7 output: {produced}"
        )
        assert any("inner_distance" in n for n in produced), (
            f"no inner-distance artifact was written: {produced}"
        )

        instances = output_dir / "instance_ids.npy"
        assert instances.is_file(), "instance_ids.npy was not written"
        restored = np.load(instances)
        assert restored.shape == (620, 900), (
            f"instance map was not restored to the original geometry: "
            f"{restored.shape}"
        )
    print(f"  predict_one_image runs and writes {len(produced)} artifacts")


def test_predict_fields_dispatch(m, model):
    """`predict_fields` returns decodable fields, tiled or not.

    `TILE_INFERENCE_ENABLED` had no reader at all before this dispatch existed.
    A 512 input must take the single pass; a larger one must take the tiled path
    and come back at source resolution.
    """
    small = np.zeros((m.IMG_SIZE, m.IMG_SIZE, 3), dtype=np.float32)
    semantic, distance, boundary = m.predict_fields(model, small)
    assert semantic.shape == (m.IMG_SIZE, m.IMG_SIZE, m.NUM_CLASSES)
    assert boundary.shape[:2] == (m.IMG_SIZE, m.IMG_SIZE)

    if m.TILE_INFERENCE_ENABLED:
        big = np.zeros((m.IMG_SIZE + 200, m.IMG_SIZE + 260, 3), dtype=np.float32)
        semantic, distance, boundary = m.predict_fields(model, big, batch_size=1)
        assert semantic.shape[:2] == big.shape[:2], (
            "tiled prediction did not return fields at source resolution"
        )
        assert distance.shape[:2] == big.shape[:2]
        assert boundary.shape[:2] == big.shape[:2]
        decoded, classes, _, scores = m.decode_instances(
            semantic, distance, boundary
        )
        assert decoded.shape == big.shape[:2], (
            "stitched fields did not decode at source resolution"
        )
        print("  predict_fields dispatches single-pass and tiled, both decodable")
    else:
        print("  predict_fields single-pass path holds (tiling disabled)")


def test_validation_previews_are_written(m, model):
    """The preview callback must actually produce files.

    It unpacked four heads after V7 went to three, so it raised on every preview
    epoch -- and the wrapper around it catches exceptions and prints a warning,
    so a run would have produced **no previews at all** while looking healthy.
    A silent diagnostic is worse than a missing one, which is why this is a test
    and not a visual check.
    """
    import tempfile

    arrays = m.load_split_arrays(
        "val", array_dir=Path(m.ARRAY_DIR)
    ) if Path(m.ARRAY_DIR).is_dir() else None
    if arrays is None:
        print("  (skipped: prepared validation arrays not present)")
        return

    with tempfile.TemporaryDirectory() as directory:
        output_dir = Path(directory) / "previews"
        callback = m.ValidationPreviewCallback(
            arrays, output_dir, preview_count=2, every_n_epochs=1,
            fail_on_error=True,
        )
        callback.set_model(model)
        callback.on_epoch_end(0, {"loss": 1.0, "val_loss": 1.1})

        per_image = sorted(output_dir.glob("epoch_*_val_*.png"))
        assert len(per_image) == 2, (
            f"expected one sheet per validation image; got {len(per_image)}"
        )
        import cv2
        sheet = cv2.imread(str(per_image[0]))
        assert sheet is not None and sheet.shape[1] == 4 * m.IMG_SIZE, (
            f"preview sheet is not a 2x4 panel grid: "
            f"{None if sheet is None else sheet.shape}"
        )
    print(f"  validation previews written, one sheet per image")


def main() -> None:
    m = load()
    m.REQUIRE_GPU = False
    m.configure_runtime()
    model = m.build_model_v7_instance()
    print(f"V7 public prediction path ({model.count_params():,} parameters)")
    test_predict_one_image_runs(m, model)
    test_predict_fields_dispatch(m, model)
    test_validation_previews_are_written(m, model)
    print("all V7 prediction contracts hold")


if __name__ == "__main__":
    main()
