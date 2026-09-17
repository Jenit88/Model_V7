# Model_v7

**Successor line to Model V6.2.** Replaces the centre-heatmap + offset-vector
instance representation with a single **normalised inner-distance field**, and
adds tiled inference for objects the 512² letterbox cannot resolve.

> **Status: trained.** 120 epochs from scratch, 11–14 Sep 2026. The released model is the
> epoch-105 checkpoint: test mask mAP50-95 **0.7554**, F1 **0.9692**. See [Results](#results).
> The sections after Results were written before the training run and are kept as the design record.

| | |
|---|---|
| Model file | `model_v7.py` |
| Parameters | 6,343,486 (V6.2: 6,417,683) |
| Input | 512×512, aspect-ratio-preserving letterbox |
| Outputs | full-res semantic · **half-res inner distance (1 ch)** · full-res instance boundary |
| Classes | `Rectangle`, `Rectangle_concave`, `circle`, `circle_full` |
| Lineage | forked from `model_v6_2.py` on 2026-09-08, when the V6.2 Tier-1 run launched |

## Results

Trained from scratch for 120 epochs. The released model is the epoch-105 checkpoint, chosen on the
complete validation split; the test split was not used for training or for choosing it.

| split | images | objects | mask mAP50-95 | mAP50 | mAP75 | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 1,048 | 48,388 | **0.8515** | 0.9874 | 0.9712 | 0.9598 | 0.9839 | **0.9717** |
| test | 432 | 33,488 | **0.7554** | 0.9731 | 0.8470 | 0.9577 | 0.9810 | **0.9692** |

Precision, recall and F1 count a mask as correct at IoU ≥ 0.50. The weakest spot on test is
`Rectangle_concave` precision, 0.50 (98 false positives against 108 objects).

- Per-class tables, training logs, evaluation reports and example predictions: [`results/`](results/README.md)
- Trained model file: [Releases](https://github.com/Jenit88/Model_V7/releases)

## Why the representation changed

The offset head asked every pixel to emit a vector to its object's centre. That
is inherently scale-dependent: a pixel on a 130 px object must emit a vector an
order of magnitude longer than one on a 16 px object, and the head was only ever
trained on the short end. **Measured on real boards, 30.1% of `Rectangle`s sit
beyond anything training produced** — and those are exactly the objects that
shatter into strips.

A normalised inner distance has no such range. It runs 0 at the rim to 1 at the
innermost pixel of every object **regardless of size**, which is the property the
offset field never had. One distance transform serves every instance: a pixel
counts as an edge if it is background or 4-adjacent to a different instance id,
so the transform measures each object's distance to its own boundary, shared
boundaries included.

What that removes, by construction rather than by threshold:

| gone | why it mattered |
|---|---|
| the 51-head-pixel assignment cap | a large object's outer pixels were unassignable however good the prediction |
| the unassigned residue, and therefore **the whole fallback path** | it produced 75 false positives against 2 true positives on validation |
| centre peaks, and `CENTER_CONFIDENCE_THRESHOLD` | its 0.10 default admitted a spurious low-confidence duplicate on top of most real pads |

## Quick start

```bash
cd /mnt/c/Users/u117134/Desktop/dev/Model_v7

./run_v7_tests.sh          # all four contract suites, CPU-only, ~3 min
./run.sh selftest          # the model's own 11-check self-test
```

Both run on CPU deliberately — these are decode, target and geometry contracts,
none of which depend on the accelerator, and the GPU should stay free.

## What is done

**Tier 2 — the head swap. Complete and wired.**

| piece | where |
|---|---|
| `build_inner_distance_target` | 1 channel of per-instance normalised distance + 1 validity mask |
| `MaskedInnerDistanceLoss` | masked Huber, rim-weighted so the boundary is not drowned by the flat interior |
| `watershed_instances` | replaces `nearest_center_assignments`; seeds from the field's own high ground |
| `build_model_v7_instance` | `center` + `offset` heads replaced by one `inner_distance` output |
| `decode_instances` | same return signature, so every call site kept working |

The model, the target generator, the loss, the compile weights and all five
`build_all_targets` call sites agree.

**Three defects the tests found, which is the point of writing them first.** The
oracle test passed at IoU 1.000 immediately and taught nothing; the degradation
test (blur + noise) earned its keep:

1. **Speckle over-segmentation** — 2 objects became 15. Every speck seeds an
   instance, and because seeds expand by nearest-core, each speck grows into a
   full-sized object the area filter cannot catch. Fixed by filtering **cores**
   before expansion.
2. **Smoothing is the wrong tool**, and it was in the first draft. A sweep showed
   sigma 1.0 buys two units of noise robustness and destroys the annulus every
   time, because a 3 px ring's core is one pixel wide. Now 0, with the sweep
   table in the constant's comment.
3. **Global thresholding was a hidden assumption** — it assumes the prediction
   preserves the normalisation, so an under-predicted object gets no core and its
   pixels are handed to its **neighbour**. A blurred 16 px disc vanished into an
   adjacent 130 px rectangle. Thresholding relative to each blob's own maximum
   fixed that.

**Tier 3 — tiled inference. Implemented and tested, not connected.**
`tile_grid`, `_tile_blend_weight` and `predict_fields_tiled` stitch overlapping
tiles by cosine taper and decode once over the result, rather than decoding each
tile and merging instances across a seam.

> **Reviewed and revised 2026-09-10.** Nine changes, validated head-to-head on real
> validation images: the instance score, the grouping rule, and the loss weight for the
> inner-distance head were all wrong. Full evidence in
> [`Change/CHANGE_REPORT.md`](Change/CHANGE_REPORT.md). The list below is what remains.

## What is not done

Read this before committing GPU time. Nothing here is speculative — each was
measured or traced in the code.

1. **The ranking signal is near-degenerate.** `decode_instances` scores each
   instance with `max(distance_full[candidate])`. But the target normalises every
   instance's peak to exactly 1.0, and `watershed_instances` guarantees every
   returned instance contains a core. Fed the **oracle** field, a 4-object scene
   decodes to scores `0.8593, 0.9863, 1.0000, 1.0000` — a 0.14 spread that comes
   from 256→512 upsampling smoothing small peaks, not from instance quality. It
   ranks by size, worst for small objects. mAP is rank-sensitive, and a
   deployment confidence floor would have nothing to bite on. **Fix this before
   training.**
2. **Tier 3 is not wired.** `predict_fields_tiled` is called only from
   `test_tier3_tiling.py`; `TILE_INFERENCE_ENABLED = True` is read nowhere.
3. **The decoders still run at half resolution** (`INSTANCE_HEAD_SIZE =
   IMG_SIZE // 2`). The `circle` assertion in `test_tier2_grouping.py:224` is
   disabled pending a resolution change — but **measured on the test split, only
   3.1% of `circle`, 3.9% of `Rectangle` and 2.3% of `circle_full` instances lack
   a usable core at head resolution, and 1–2% are degenerate.** The 1.5 px annulus
   in the synthetic test is a worst case, not the common one, so Tier 3 is worth a
   few per cent of objects rather than the step change it was scoped as. Plan
   against that number.
4. **The recorded training config misdescribes the model.** It still emits
   `"offset": {"name": "MaskedOffsetPixelHuberLoss"}` and lists `center`/`offset`
   in `head_weights` while omitting `inner_distance`. The actual `compile()` is
   correct — only the record is wrong.
5. **The V5-transfer validator was broken by a blind rename** — it demands an
   `inner_distance` layer from a V5 source, while the comment above it still
   talks about centre and offset heads. Dead path (scratch-init only), but
   unreviewed.
6. **`minimum_core_area` filters globally, not per blob.** If any core clears the
   floor, every sub-floor core is dropped; a blob that loses its core has its
   pixels handed to the nearest *other* core, because expansion is Euclidean with
   no connectivity constraint and the old 51 px cap is gone. That is defect 3
   above returning through a different door, and it can produce spatially
   disconnected masks.

## Tests

```bash
./run_v7_tests.sh
```

| suite | what it holds |
|---|---|
| `run_selftest_v7` | the model's own 11 contracts: decode, targets, ranking, augmentation, the build |
| `test_tier2_grouping` | grouping under blur + noise, across scale extremes, touching pads, thin annuli, frame truncation, dense small pads |
| `test_tier2_loss` | the masked, rim-weighted Huber |
| `test_tier3_tiling` | tile coverage and seams, including a real forward pass |

Seams matter more than they look: a visible one is read by the decoder as a
boundary, so it manufactures instance splits exactly where two tiles meet —
which would look like an over-segmentation bug anywhere except its actual cause.

`sweep_tier2_params.py` sweeps the grouping constants over the same scenes.

## Training

Same measured laptop configuration as V6.2 — batch 2 × 2 accumulation,
`mixed_float16`, cosine LR 3e-4, ~35 min/epoch on the RTX PRO 1000 8 GB.

```bash
./run.sh selftest              # ~1 min, no dataset/GPU needed
./session.sh start prepare     # one-off, if the arrays do not exist yet
./session.sh start scratch     # the training run
./session.sh attach            # watch it   (detach: Ctrl-b then d)
```

**Verify the GPU is real before starting**, because TensorFlow falls back to the
CPU silently here:

```bash
source ./env.sh
$PCB_PYTHON -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Full environment, dataset layout, crash recovery, evaluation, threshold fitting
and deployment work exactly as documented in the V6.2 repository's README — this
project is a copy of that infrastructure. The differences are below.

### This project keeps its own identity, so the two never collide

| | V6.2 project | this project |
|---|---|---|
| launcher | `train_rtx.py` | **`train_v7.py`** |
| tmux session | `pcb` | **`v7`** |
| supervisor | `pcb_supervisor.sh` | **`v7_supervisor.sh`** |
| logon task | `PCB-v62-training-supervisor` | **`Model-v7-training-supervisor`** |
| supervisor's wrapper session | `pcbsup` | **`v7sup`** |
| default output | `~/Models/Model_v6_2_scratch_rtx` | **`~/Models/Model_v7_scratch_rtx`** |

That separation is not cosmetic. The watchdog and supervisor identify the job by
matching its process line; with both projects using `train_rtx.py`, one
project's watchdog would answer to the other's run. Prepared arrays are shared
(`~/data/pcb_v62_arrays_repaired`) because the array format is unchanged — the
inner-distance target is built from the same semantic and instance rasters.

> tmux resolves a target session by **prefix**, so `-t v7` also matches `v7sup`.
> Every target in `session.sh`, `watchdog.sh` and `v7_supervisor.sh` uses `=v7`
> to force an exact match. Removing that makes a logon resume silently do
> nothing — it happened on 10-09 in the V6.2 project.

## Layout

```
model_v7.py             the model — every stage lives here
train_v7.py             measured laptop configuration + benchmark harness
run.sh                  single entry point for every mode
env.sh                  environment; MUST be sourced before any Python
session.sh              detached tmux session 'v7'
watchdog.sh             restart on process death (inside WSL)
v7_supervisor.sh        restart after host reboot (Windows logon task)
register_v7_watchdog.ps1
active_run.env          which run the supervisors keep alive

run_v7_tests.sh         all four contract suites
run_selftest_v7.py      the model's own self-test, on CPU
test_tier2_grouping.py  watershed grouping under degradation
test_tier2_loss.py      the inner-distance loss
test_tier3_tiling.py    tile coverage and seams
sweep_tier2_params.py   grouping constant sweep

tta.py                  dihedral TTA + algebra self-check
tta_v7.py               the V7 field conventions for TTA
deployment.py           the real-board operating point
reevaluate.py           re-score a checkpoint with the current decoder
fit_thresholds.py       split-validated decoder threshold search
predict_folder.py       sample a folder, report the score distribution
predict_pictures.py     overlays only, resumable
dashboard.py            live dashboard on :8088

results/README.md             training outputs, evaluation reports, example predictions
results/threshold_fit.json    fitted thresholds, inherited from V6.2
memory/                       the dataset audit and the uplift plan
```

## Where to start

In order, and the first one is not optional:

1. **Fix the ranking signal.** Training against a score that cannot order
   detections wastes the run — mAP is computed from that ordering.
2. **Wire tiled inference**, or delete `TILE_INFERENCE_ENABLED` so it stops
   claiming to be on.
3. **Decide the resolution question.** Full-resolution decoders and tiling
   address the same defect from two directions; `circle` needs one of them.
4. Correct the recorded training config, then train.

## Caveats inherited from the V6.2 measurements

1. 44 images are byte-identical between val and test (10% of test), and decoder
   thresholds are fitted on val.
2. Test is not train: objects are ~0.6× the diameter and 2.4× as dense.
3. `Rectangle_concave` has 680 training polygons. No architecture change
   substitutes for more labels.
4. `results/threshold_fit.json` was fitted for the **centre/offset** decoder.
   All five constants still exist in `model_v7.py` — they are kept for the
   retained `decode_instances_centre_offset` path — so `deployment.py` applies
   them without error. But `center_confidence` and `center_nms_radius` are
   **inert** for the V7 decoder, which has no centre head, and the remaining
   three were tuned against a different grouping rule. Refit on validation
   before trusting any of them, and treat a silent success here as a warning.
