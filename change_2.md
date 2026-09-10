# Model_v7 — change report, round 2

**Date:** 2026-09-10 · **Trigger:** `inspection/MODEL_V7_INSPECTION_REPORT.md` plus four
follow-up requests on training, threshold fitting, sampling and decoder calibration.
**Result:** 6/6 contract suites pass, including a new end-to-end prediction test on a real
model. **Status:** still untrained.

Round 1 is `CHANGE_REPORT.md`. This covers everything after it.

---

## Part A — the inspection's findings, verified one by one

I checked each claim against the source before acting on any of it. **Every code-level finding
was true.** Two were release-blocking.

| # | Finding | Verified? | Action |
|---|---|---|---|
| P0 | `predict_one_image` references a deleted centre head | **TRUE** — `model_v7.py:18221` used `center_probabilities_small`, unbound in that function | **Fixed** |
| P0 | `fit_thresholds.py` is still a V6 four-head program | **TRUE** — unpacked 4 outputs, searched `center_confidence`/`center_nms_radius` | **Rewritten** |
| P1 | Tiled inference not reachable from public prediction | **TRUE** | **Fixed + tested** |
| P1 | Benchmark targets a missing script | **TRUE** — `train_v7.py:187` still called `train_rtx.py` | **Fixed** |
| P1 | Deployment settings inherited from V6 | **TRUE** | **Fixed** — V6 file quarantined |
| P2 | Dashboard plots centre/offset, hides inner-distance | **TRUE** — `dashboard.py:29-30` | **Fixed** |
| P2 | `./run.sh predict-folder` documented but not exposed | **TRUE** | **Fixed** |
| P2 | README status claims stale | **TRUE, and mine** | **Fixed** |
| — | Parameter counts, V6.2 baseline figures, 44 val/test duplicates, no V7 checkpoint | **TRUE** — independently reproduced | Adopted |

**The P0 prediction bug deserves emphasis.** It was invisible to every component test, because
each one tested a function in isolation and the break was in the glue. That is exactly the gap
`test_v7_predict.py` now closes.

### What was fixed, precisely

**1. `predict_one_image` (P0).** The function correctly unpacked `semantic, inner_distance,
boundary`, then 30 lines later used `center_probabilities_small` — a `NameError` on every real
prediction, reached by `predict_folder.py` and `predict_pictures.py` too. The centre heatmap
artifact is now the **inner-distance field**, V7's actual per-pixel instance evidence:
`centres.png` → `inner_distance.png`, one channel instead of `NUM_CLASSES - 1`, attenuated by
foreground probability instead of maxed over per-class centre maps. The misleading local
`instance_center_scores` was renamed `instance_scores`.

A second, real defect surfaced *while testing the fix*: `restore_map_to_original` drops a
trailing singleton axis, so the restored field came back 2-D and the new shape contract
rejected it. Caught by the smoke test, not by reading.

**2. `fit_thresholds.py` (P0).** Now unpacks three outputs, and searches **V7's vocabulary**:

| removed (inert in V7) | added |
|---|---|
| `center_confidence` | `inner_distance_core_threshold` |
| `center_nms_radius` | `minimum_core_area` |
| | `minimum_confidence` — the component-consistency floor |

`semantic_confidence`, `minimum_instance_area` and `boundary_confidence` are retained. The
consistency floor is applied inside the scorer via `filter_instances_by_confidence`, or the
dimension would have been inert — the same defect the V6 file had.

**3. The V6 threshold file is no longer reachable.** `results/threshold_fit.json` →
`results/threshold_fit_v6_legacy.json`; `deployment.py` reads only `threshold_fit_v7.json` and
warns explicitly when it is missing *and* a legacy file is present. Two of the V6 file's five
values set attributes V7's decoder never reads and the other three were fitted against a
different grouping rule — applying it would have succeeded silently and calibrated nothing,
which is the worst failure mode because it looks configured.

**4. Tiling reachable and proven.** `test_v7_predict.py` asserts `predict_fields` takes the
single pass at 512 and the tiled path above it, returns fields at source resolution, and that
those stitched fields **decode** at source resolution.

---

## Part B — sampling: the finding that overturned an assumption

Asked whether stronger small/dense sampling matters, I measured the pipeline's own output
instead of reasoning about it. Median over 48–60 real images:

| sample | obj/img | Rectangle | circle | circle_full |
|---|---:|---:|---:|---:|
| train, no augmentation | 17.0 | 37.2 | 33.5 | 21.3 |
| **train, augmented (current)** | **14.5** | **41.1** | **43.5** | **26.2** |
| **TEST (what it is scored on)** | **75.0** | **22.1** | **16.0** | **15.9** |

**The augmentation moves away from the test distribution on both axes** — fewer objects and
larger ones, where test needs many and smaller. V5 zoom-to-fill enlarges on every rotation.
This is a five-fold density gap and a two-fold scale gap, in the wrong direction.

**Zoom cannot fix it.** A sweep of zoom-out probability (0.50 → 0.85) and ladder depth
(50% → 70%) moved median Rectangle diameter 41.1 → 35.7 and left `obj/img` at 14.5 in *every*
configuration — because zooming out shrinks the objects in a frame but never adds any. Object
count is a property of the source board. Copy-paste adds at most `COPY_PASTE_MAX_OBJECTS = 3`,
so it cannot bridge 17 → 75 either.

**Mosaic can, and does.** A 2×2 tile of four independently augmented samples multiplies count
by four and halves diameters in one operation:

| sample | obj/img | Rectangle | circle | circle_full | distance from test |
|---|---:|---:|---:|---:|---:|
| TEST (target) | 78.0 | 21.9 | 16.0 | 15.9 | 0 |
| current augmented | 20.5 | 42.4 | 40.9 | 28.9 | **2.068** |
| **+ mosaic** | **93.0** | **19.0** | **20.6** | **12.2** | **0.400** |

An **81% reduction** in distribution gap, on all four statistics at once.

`mosaic_samples()` downscales images by INTER_AREA and label maps by nearest neighbour
(an interpolated class or instance id is meaningless), **re-bases instance ids per quadrant** so
two objects carrying id 1 in different source frames cannot merge, and drops objects that fall
below `MIN_AUGMENTED_INSTANCE_AREA` after halving — removing them from the semantic map too,
because a label too small to be a valid target must not linger as a phantom. All three
properties are asserted in `test_v7_decode.py`.

### It ships **off** by default, and that is the point

`MOSAIC_PROBABILITY = 0.00`.

The V6.2 Tier-1 run was an augmentation change argued from exactly this kind of
distribution reasoning, and it measurably cost mask precision. Any resampling trades image
fidelity for coverage. Mosaic downscales by two — unlike zoom-in it *discards* detail rather
than inventing it, which is the direction test objects genuinely occupy, so the risk is lower,
not absent.

**Recommended first experiment: `MOSAIC_PROBABILITY = 0.25`**, A/B against 0.00 on a short run.
A mixture is likely better than a match: validation sits at ~16 objects/image and test at ~78,
so a model must handle both, and p = 1.0 slightly overshoots test density (93 vs 78).

---

## Part C — training and calibration: what I did, and what I declined

### The loss-weight sweep — a cheaper design

The request was to test weights 5, 10 and 15 across multiple seeds. At ~35 min/epoch and 120
epochs, one run is 2.8 days; three weights × three seeds is **~25 days of GPU**. That budget
does not exist on one 8 GB laptop.

**The same question is answerable in about 18 hours.** V6.2 reached mask mAP50-95 of 0.7284 by
**epoch 10**, so the signal that separates a healthy head from a starved one is present early.
Three 10–12 epoch runs at weights 5/10/15, compared on epoch-10 instance mAP *and* on the
inner-distance loss trajectory, answer it at 1/30th the cost. Only the winner needs a full run.

Two things make that comparison readable, and both are now in place:

* **The dashboard plots the inner-distance loss** instead of the two heads V7 deleted. It was
  showing empty centre/offset panels and hiding the only curve that says whether instance
  separation is being learned at all.
* **`training_log.csv` already logs each head separately**, so the loss *share* — the quantity
  the weight actually controls — can be computed per epoch rather than inferred.

**On seeds:** this project already has a better instrument than seed repetition for its main
question. Step 7 compared V6.2 against YOLO with a **1,000-draw paired bootstrap** over images,
which gives a confidence interval from a single run pair. That captures evaluation variance —
the larger term at 432 images — though not initialisation variance. Report the paired-bootstrap
interval from one seed, say plainly that seed variance is unmeasured, and spend a second seed
only at the winning configuration.

### Clean validation — implemented as far as it can be without a model

The inspection is right that selecting checkpoints on contaminated validation biases the test
result. I recomputed the duplicates directly rather than trusting the audit:

```
val images: 1048   test images: 432
val images byte-identical to a test image: 44  (10.2% of test)
clean validation size: 1004 images
```

Exactly the audited 44. The index list is now at
**`results/validation_test_duplicates.json`**, ready to exclude when selecting checkpoints,
fitting thresholds, or quoting a test number. Wiring it into the selection callback is a
training-path change and is the natural next step.

### Per-class and size-stratified thresholds — deliberately **not** done

The request was per-class thresholds and separate core thresholds / minimum areas for small and
large objects. I think that is the wrong move right now, for three reasons:

1. **There is no V7 model.** Every one of those parameters is fitted against predictions. There
   is nothing to fit against, and nothing to validate a fit with.
2. **It multiplies overfitting exactly where this project is most exposed.** Six shared
   dimensions become roughly twenty-four when split four ways by class, fitted on 1,048
   validation images of which **10.2% are byte-identical to test**. The existing V6 fit is the
   cautionary case: it won on the fitting half, survived its held-out half, and still scored
   *worse* on test (0.7710 → 0.7694), because it bought precision with recall.
3. **The infrastructure that makes the decision answerable is the cheaper thing to build first**
   — the duplicate list above, per-head loss visibility, and size-stratified reporting.

`fit_thresholds.py` keeps the split-validated design that already exists: fit on half of
validation, re-score the winner on the held-out half, and refuse a fit that does not survive.
Add class and size dimensions only when a trained model shows a class or size band failing for
a reason a shared threshold cannot express.

### Validating the consistency score against false splits and merges

Partly done, and reported honestly. `test_v7_decode.py` asserts the score ranks **every**
correct object above **every** wrong grouping on a synthetic scene covering a fragment, a
quarter, a rim sliver and a merged pair — worst correct 0.901 against best wrong 0.864. Round 1
also validated it end-to-end on real validation images across a degradation ladder, where it
contributed **+0.0060 mAP50-95** at harsh degradation.

What is still missing, and cannot be supplied yet: validation against **real** false splits and
merges produced by a trained model. Synthetic constructs and blurred oracle fields are not the
same as a network's structured errors.

---

## Files changed in round 2

```
model_v7.py            predict_one_image repaired (P0) + inner-distance artifact
                       + shape contract fix + mosaic constants and mosaic_samples()
                       + loader integration
fit_thresholds.py      V7 three-head schema, V7 search vocabulary, consistency-score
                       dimension, writes threshold_fit_v7.json
deployment.py          V7-only threshold vocabulary; refuses the V6 fit and says so
dashboard.py           inner-distance loss replaces centre/offset panels
train_v7.py            benchmark subprocess targets train_v7.py
run.sh                 predict-folder exposed; usage line corrected
run_v7_tests.sh        registers test_v7_decode and test_v7_predict
test_v7_predict.py     new — end-to-end prediction on a real model
test_v7_decode.py      + mosaic label-safety contract
README.md              stale status items corrected
results/threshold_fit.json -> threshold_fit_v6_legacy.json   (quarantined)
results/validation_test_duplicates.json                      (new)
```

## Where this leaves the inspection's gates

| Gate | Status |
|---|---|
| 1 — repair P0 prediction interface | **Done**, proven by `test_v7_predict.py` on a real model |
| 2 — repair P0 calibration interface | **Done** — V7 schema and vocabulary; unrunnable until a checkpoint exists |
| 3 — native-resolution route | **Reachable and tested**; `predict_one_image` still letterboxes to 512 before predicting, so a board larger than one tile is not yet routed through `predict_fields` |
| 4 — restore benchmark path | **Done** |
| 5 — dashboard V7 fields | **Done** |
| 6 — short smoke campaign | Not started — needs GPU time |
| 7 — controlled full campaign | Not started |
| 8 — unbiased comparison | Not started; the duplicate list is the missing input, now available |

**Gate 3 is honestly partial.** The dispatch exists, is tested, and decodes at source
resolution, but `predict_one_image` letterboxes first, so full-resolution boards still take the
single pass. Completing it means restructuring that function to predict on the original image
and skip the geometry restore — worth doing, and worth doing separately from this batch.

## The claim to keep making, and the one not to

Unchanged from round 1: **V7 is a well-founded, now-runnable research candidate.** Its
contracts pass, its public prediction path works, its decode changes are validated on real data,
and its training-distribution gap is measured and has a measured remedy.

It is still **untrained**. Nothing here supports a claim that V7 is more accurate than V6.2.
