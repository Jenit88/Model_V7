# Model_v7 — change report

**Date:** 2026-09-10 · **Base:** `model_v6_4.py` as forked from `model_v6_2.py` on 2026-09-08
· **Result:** 5/5 contract suites pass · **Status:** still untrained — every number below is a
decode-path or loss-scale measurement, not a training result.

Everything here was measured on this machine. Where a claim rests on a number, the script
that produced it is in `Change/evidence/`, and `Change/evidence/model_before_changes.py` is
the exact file the "old" column refers to.

---

## Summary

Nine changes. Two are substantive and validated head-to-head on real data; one is a loss-scale
defect that would have quietly under-trained the model's only instance-separating head; the
rest are coherence fixes and one augmentation reversal argued from the Tier-1 result.

| # | Change | Kind | Evidence |
|---|---|---|---|
| 1 | Instance score replaced with field self-consistency | **correctness** | +0.0060 mAP50-95 at harsh degradation |
| 2 | `watershed_instances` works per foreground blob | **correctness** | +0.0028 mAP50-95, affects 2–4% of real objects |
| 3 | `INNER_DISTANCE_LOSS_WEIGHT` 1.00 → 10.00 | **correctness** | head held 3.8% of loss budget, should hold ~29% |
| 4 | Per-component scoring in `add_partitioned_instances` | correctness | fragments no longer inherit a real object's confidence |
| 5 | `ZOOM_IN_PROBABILITY` 0.30 → 0.00 | tuning | Tier-1 measured −0.003…−0.013 mAP50-95 |
| 6 | `predict_fields()` dispatch added | wiring | `TILE_INFERENCE_ENABLED` had no reader |
| 7 | Recorded training config corrected | coherence | named heads the model no longer has |
| 8 | V5-transfer validator repaired | coherence | demanded a layer no V5 model has |
| 9 | `test_v7_decode.py` added | regression | 4 contracts, wired into `run_v7_tests.sh` |

**Combined effect on the decode path, real validation images, all heads oracle except the
distance field:**

| field degradation | old mAP50-95 | new mAP50-95 | delta | decode error removed |
|---|---:|---:|---:|---:|
| oracle | 0.9905 | 0.9950 | **+0.0045** | 47% |
| mild (σ1.0, n0.02) | 0.9934 | 0.9950 | +0.0015 | 24% |
| moderate (σ2.0, n0.05) | 0.9896 | 0.9923 | +0.0027 | 26% |
| harsh (σ3.0, n0.08) | 0.9807 | 0.9894 | **+0.0088** | 45% |

The gain grows as the field degrades — which is the signature of fixing failures that only
appear when the prediction is imperfect. A trained model produces an imperfect field, so the
harsh row is the one that matters.

---

## 1. The instance score was degenerate — replaced

### What was wrong

`decode_instances` scored every instance with `max(distance_full[candidate])`. But the target
normalises each instance's inner distance to peak at **exactly 1.0**, and `watershed_instances`
guarantees every returned instance contains a core. So the peak is ~1.0 for a correct object,
a merged pair and a rim sliver alike.

Fed the **oracle** field — a perfect prediction — a four-object scene decoded to:

```
scores: 0.8593, 0.9863, 1.0000, 1.0000     spread 0.1407
```

That 0.14 of spread comes from bilinear 256→512 upsampling smoothing small peaks, not from
instance quality. It ranks by object *size*, and ranks small objects **worst** — backwards for
a dataset whose hard classes are ~16 px. Mask AP is computed from that ordering, and the
deployment confidence floor has nothing to bite on.

There was a second-order version of the same bug: `boundary_partition` can split one candidate
into several instances, and all of them inherited a single confidence — so a rim fragment was
emitted carrying exactly the confidence of the real object beside it.

### What replaced it

`inner_distance_instance_score()` — a self-consistency measure. A normalised inner-distance
field is fully determined by the shape it describes, so given a decoded mask, the field a
correct prediction *must* produce can be computed exactly. Score the agreement:

```
implied  = distanceTransform(mask) / its own max
score    = (1 − mean|predicted − implied|) × mean semantic confidence
```

A wrong grouping cannot fake this:

* a **correct object** — prediction equals its own implied field;
* a **fragment** — carries a field normalised over the *whole* object, which never reaches 1.0
  inside the fragment and has the wrong profile for the fragment's shape;
* a **merged pair** — two humps where one broad hump is implied;
* a **rim sliver** — a field near zero where its own shape implies a full 0→1 ramp.

**One refinement, measured.** The prediction comes from a half-resolution head and is
bilinearly upsampled, so a crisp analytic transform is *not* what a correct prediction looks
like — and the mismatch falls hardest on small objects, exactly the ones that must not be
under-ranked. Putting the implied field through the same resolution round trip lifted a 16 px
disc from 0.890 to 0.901 while leaving every wrong grouping where it was:

| construct | naive | resolution-matched |
|---|---:|---:|
| 130 px rectangle | 0.982 | 0.982 |
| **16 px disc** | 0.890 | **0.901** |
| touching pad A / B | 0.980 | 0.979 |
| fragment — half a rectangle | 0.757 | 0.757 |
| fragment — quarter | 0.770 | 0.770 |
| fragment — rim sliver | 0.530 | 0.545 |
| wrong — merged pair | 0.865 | 0.864 |
| **margin (worst correct − best wrong)** | **+0.025** | **+0.037** |

A 48% wider margin for two small resizes on a crop.

### Cost

Two resizes and one distance transform per instance, all on the component's own bounding box —
cost scales with object area, not image area. Measured decode time over 40 real validation
images: **6.6 s → 7.2 s**, about 9%.

*Evidence: `evidence/probe_v7_score.py`.*

---

## 2. `watershed_instances` now works per foreground blob

### What was wrong

Two operations were global across the whole class mask:

* **The minimum-core-area filter.** If any core anywhere cleared the floor, every sub-floor
  core was dropped. A small object beside a large one lost its core while the large one kept
  its own.
* **Expansion.** `distanceTransformWithLabels` ran once over the whole image, Euclidean, with
  no connectivity constraint and no distance cap (the old 51 px cap is deliberately gone). A
  blob left without a core did not become its own instance — its pixels went to the nearest
  *other* blob's core, producing a spatially disconnected mask.

Together these are the "16 px disc vanishes into the 130 px rectangle" failure that relative
thresholding was written to fix, returning through a different door.

### How common — measured, not assumed

Instances whose core falls below `MIN_INNER_DISTANCE_CORE_AREA` at head resolution:

| class | test split | val split |
|---|---:|---:|
| `Rectangle` | 3.9% | 2.1% |
| `circle` | 3.1% | 0.4% |
| `circle_full` | 2.3% | 2.3% |
| `Rectangle_concave` | 0.0% | 0.0% |

At 77.5 objects per test image that is **roughly two objects per image** at risk of being
absorbed into a neighbour. Not a corner case.

### What changed

Cores are found, closed, filtered and expanded **inside each connected foreground blob**,
cropped to its bounding box. If the area filter would empty a blob, its largest core is kept —
an object whose core is genuinely tiny is still an object and must not be handed to a
neighbour. A blob with no core at all becomes one instance, which under-segments rather than
losing the object.

The docstring gained a fourth guarantee: **no cross-object leakage.**

---

## 3. The loss weight was inherited from the wrong head

**This is the highest-impact change in the report, and the one least visible by reading.**

`INNER_DISTANCE_LOSS_WEIGHT = 1.00` was carried over from `OFFSET_LOSS_WEIGHT = 1.00`. But the
offset loss was never what trained instance geometry.

V6.2's actual loss budget, from the Tier-1 run's `training_log.csv` at epoch 73. The arithmetic
reconstructs the logged total of **0.17761 exactly**, so these shares are exact:

| head | raw | × weight | contribution | share |
|---|---:|---:|---:|---:|
| semantic | 0.0751 | 1.00 | 0.0751 | 42.3% |
| **centre** | 0.1907 | 0.25 | 0.0477 | **26.8%** |
| **offset** | 0.0040 | 1.00 | 0.0040 | **2.3%** |
| boundary | 0.1694 | 0.30 | 0.0508 | 28.6% |
| | | | **0.1776** | instance geometry **29.1%** |

The **centre head carried instance-geometry learning**, not the offset head. V7 deletes the
centre head and keeps offset's weight.

`MaskedInnerDistanceLoss` measured on real targets — a Huber on a `[0,1]` quantity with
δ = 0.10 is simply an order of magnitude smaller than the losses beside it:

| prediction state | loss |
|---|---:|
| initialisation (sigmoid ≈ 0.5) | 0.0240 |
| mean abs error 0.20 | 0.0118 |
| **mean abs error 0.10 (mid-training)** | **0.0050** |
| mean abs error 0.05 | 0.0016 |
| perfect | 0.0000 |

At weight 1.00 the only head that separates instances would hold **3.8%** of the budget where
its predecessors held 29.1% — an **8× cut in the gradient signal driving instance separation**,
introduced silently by a constant that looked untouched.

| weight | contribution at mid-training | share |
|---:|---:|---:|
| 1.0 | 0.0050 | 3.8% |
| 6.0 | 0.0300 | 19.2% |
| 8.0 | 0.0400 | 24.1% |
| **10.0** | **0.0500** | **28.4%** |
| 15.0 | 0.0750 | 37.3% |

Set to **10.00**, matching the 29.1% the replaced heads held. A regression test pins the share
to the 18–36% band.

> **The caveat that must travel with this.** The value is calibrated at the mid-training
> operating point. A dense regression loss decays toward zero as it fits, while cross-entropy
> plateaus at a nonzero floor, so this share is **not constant across a run**. It is one
> constant and the single most likely thing to need revisiting once a run exists. At
> initialisation (0.024 raw) weight 10 gives 0.24, well below the early semantic loss, so it
> will not dominate the warmup.

*Evidence: `evidence/loss_scale.py`.*

---

## 4–9. The rest

**4. Per-component scoring.** `add_partitioned_instances` takes a `score_fn` and calls it per
written component instead of assigning one candidate-level confidence to all of them.

**5. `ZOOM_IN_PROBABILITY` 0.30 → 0.00.** Argued from both directions:

* *The reason for it is gone.* The ladder existed because the offset head emitted a vector
  proportional to object size and 30.1% of real-board Rectangles sat beyond training's range.
  A normalised inner distance is bounded in [0,1] at every scale; that defect cannot occur here.
* *And it was measured to cost.* The Tier-1 run **is** this ladder against the 05-09 baseline.
  Over 15 shared evaluations: mask mAP50-95 **−0.003 to −0.013**, semantic foreground mIoU
  **−0.0048**, while precision **+0.010**, F1 **+0.004** and mAP50 **+0.001** — better at
  finding objects, worse at outlining them. Per class the benefit was monotone in object size
  (+0.0235 F1 on the ~108 px class, +0.0001 on the ~16 px ones). Zoom-in upsamples: it cannot
  create detail, and a nearest-neighbour-resampled label boundary is a staircase, so 15% of
  every epoch trained the boundary head on manufactured edges.

Zoom-*out* keeps its p = 0.50 — it crops nothing and invents nothing, and the audit shows the
test split is more zoomed-out than train, not less.

**6. `predict_fields()` dispatch.** `TILE_INFERENCE_ENABLED = True` had no reader;
`predict_fields_tiled` was called only from its own test. The new entry point dispatches on the
flag *and* on whether the source is larger than one tile — tiling a 512² letterboxed frame
would run the network once and call it a grid. The prepared arrays **are** 512², so benchmark
evaluation is unaffected; only prediction from original-resolution boards changes.

> **Expect a few per cent, not a step change.** Tiling exists because a thin ring cannot be
> resolved at head resolution — but measured on the test split, only 3.1% of `circle`, 3.9% of
> `Rectangle` and 2.3% of `circle_full` have a sub-floor core, and 1–2% are degenerate. **The
> synthetic 1.5 px annulus that motivated Tier 3 is a worst case, not the common one.** This
> materially lowers Tier 3's expected value and is the most useful thing the review found about
> it.

**7. Recorded training config.** It emitted `"offset": {"name": "MaskedOffsetPixelHuberLoss"}`
and listed `center`/`offset` in `head_weights` while omitting `inner_distance`. Now mirrors
`compile_model`'s actual `loss_weights`.

**8. V5-transfer validator.** A rename had swept `inner_distance` into its expected-outputs
dict, so it demanded a head no V5 model has ever had and the path could not run. Now checks
`semantic` and `boundary` only — this path copies compatible encoder blocks, not heads.

**9. `test_v7_decode.py`** — four contracts, added to `run_v7_tests.sh`:
score separation, small-object survival, instance connectivity, loss-budget share.

---

## Validation

### Ablation — which change earns the gain

Real validation images, semantic and boundary held at oracle so the comparison isolates the
grouping and the score.

| degradation | old + old | **new grouping** + old score | new grouping + **new score** | grouping | score |
|---|---:|---:|---:|---:|---:|
| moderate | 0.9896 | 0.9922 | 0.9923 | **+0.0025** | +0.0001 |
| harsh | 0.9807 | 0.9834 | 0.9894 | **+0.0028** | **+0.0060** |

The grouping fix contributes a steady ~+0.0027 at both levels. The score fix contributes almost
nothing at moderate and **+0.0060 at harsh** — exactly as expected, since a ranking signal only
matters when there are wrong detections to push down. A trained model, especially early or on
the denser test split, sits nearer "harsh" than "oracle".

### Realism check — do the gains survive imperfect heads?

Degrading semantic and boundary as well, not just the distance field:

| setting | old | new | delta | decode error removed |
|---|---:|---:|---:|---:|
| all heads degraded, moderate | 0.9913 | 0.9940 | +0.0027 | 31% |
| all heads degraded, harsh | 0.9482 | 0.9528 | +0.0047 | 9% |

The gains survive, so they are not an artifact of the oracle setup. At harsh the *share*
removed falls because most remaining error is then attributable to the semantic head rather
than the decoder.

### What this validation does **not** establish

Stated plainly, because the absolute numbers look better than they are:

1. **Absolute mAP here (0.98–0.99) is inflated** by oracle semantic and boundary. A trained
   V6.2 scores ~0.83 on real validation. The deltas are meaningful; the levels are not, and the
   deltas will not transfer 1:1 into a trained run.
2. **The degradation is synthetic** — Gaussian blur plus i.i.d. noise. A real model's errors are
   structured and correlated, not i.i.d.
3. **No V7 model has been trained.** Everything is decode-path and loss-scale evidence.
4. The **loss weight is the one change with no end-to-end validation at all** — it cannot have
   any until a run exists.

---

## Files changed

```
model_v7.py          9 edits: score function added and wired, watershed rewritten per blob,
                     predict_fields dispatch added, loss weight, zoom-in probability,
                     config record, V5 validator, two docstrings
run_v7_tests.sh      registers test_v7_decode
test_v7_decode.py    new — four regression contracts
```

Unchanged and deliberately so: the target generator, the loss's own maths, the tiling geometry,
and every V6.2-compatibility path.

## Where to go next

1. **Train it.** Every open question now needs a run, and the loss weight most of all.
2. **Re-check the loss share at epochs 5, 20, 50** against `training_log.csv`. If the
   inner-distance contribution collapses as it fits, the weight — not the architecture — is
   what to adjust.
3. **Treat Tier 3 as a few-per-cent improvement**, not a step change. The 2–4% measurement
   above is the number to plan against.
4. **Score the Tier-1 checkpoint on real boards** before concluding zoom-in was wrong
   everywhere. It lost on the benchmark; the distribution it was built for was never measured.
