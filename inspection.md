# Model V7 Inspection Report

## Executive assessment

Model V7 is a materially redesigned, untrained instance-segmentation implementation with a sound technical direction, a coherent three-head tensor contract, and a passing component-level test suite. Its most important change is the replacement of V6.2's class-centre and offset-vector instance heads with a normalised per-instance inner-distance field, decoded through classwise watershed-style partitioning.

The design is promising for this PCB dataset because the known difficult cases are small, dense, closely adjacent components and boundary-sensitive circles. The V6.2 reference model has excellent loose-overlap results but materially weaker tight-overlap performance, especially for the circle class. V7's target and decoder are aimed at exactly that weakness.

That is not yet evidence that V7 is better. There is no trained V7 checkpoint, no V7 validation/test evaluation, and no valid V7 threshold-fit result. The current source also contains two release-blocking legacy API remnants: the public single-image prediction function refers to a deleted centre head, and the threshold-fitting program still expects V6's four outputs. Consequently, the present verdict is:

| Area | Inspection result | Decision |
|---|---|---|
| Model architecture and head contract | Internally coherent; independently tested | Ready for controlled smoke testing |
| Scratch training path | Present and configured for V7 | Conditionally usable |
| V7 trained-model evidence | No V7 output directory or checkpoint found | Not established |
| Public prediction and calibration path | Broken by V6 centre/offset remnants | Not operationally ready |
| Native-resolution tiled inference | Component exists but is not reached by public prediction | Not operationally ready |
| Deployment threshold profile | Inherited V6 settings, not V7-calibrated | Do not deploy |
| Claim that V7 outperforms V6.2 | Unsupported at this time | Do not make this claim |

The highest-value next action is to correct and test the two P0 interface mismatches before committing a full 120-epoch run. A small scratch smoke run can still be useful to verify loss flow and artifact writing, but it would not make the repository inference-ready.

## Scope, method, and integrity statement

This was a read-only technical inspection of:

- Model V7 source, scripts, tests, configuration, existing results, change evidence, and dataset-audit notes under C:\Users\u117134\Desktop\dev\Model_v7.
- The V6.2, V6.3, and available V6.4 source snapshots under C:\Users\u117134\Desktop\dev\model-v6-2.
- The trained V6.2 scratch artifact at /home/u117134c/Models/Model_v6_2_scratch_rtx.
- Relevant scientific literature on learned watershed instance segmentation and multi-scale weighted feature fusion.

The source snapshot used for this report has the following SHA-256 identities:

| File | SHA-256 |
|---|---|
| Model_v7/model_v7.py | BB69F826CAA51DEA2A0F156C6CE19CD241D0FF1EAC19DAC09EA5786926ECDBAC |
| model-v6-2/model_v6_2.py | 099D5EE31B5847A72E03B690633AF3A7C2485C4DA8CE7E6483BA4CDBC7F7EDA9 |
| model-v6-2/model_v6_3.py | 1ECB7EEC1FFAF26723BD7954D2DBE2DD722CC7897E065E3001E93C72987F86E3 |

I verified that /home/u117134c/Models/Model_v7_scratch_rtx does not exist at inspection time. This is consistent with the V7 README statement that V7 has not yet been trained and has no V7 checkpoint.

The following V7 contract suite was executed with GPU visibility disabled and bytecode writing disabled. It passed without modifying source code or model artifacts:

~~~text
run_selftest_v7            PASS
test_tier2_grouping        PASS
test_tier2_loss            PASS
test_tier3_tiling          PASS
test_v7_decode             PASS

all V7 contracts hold
~~~

This report is the only new filesystem artifact created by this inspection. No source code, configuration, weights, checkpoint, data array, or existing result was changed.

## Architecture and task contract

### Task and output schema

V7 is a four-foreground-class instance segmentation system for Rectangle, Rectangle_concave, circle, and circle_full, plus background. Inputs are letterboxed to 512 by 512 pixels in the standard path.

The core output contract is:

| Head | V6.2 | V7 | Purpose |
|---|---|---|---|
| semantic | [B, 512, 512, 5] | [B, 512, 512, 5] | Background and foreground class probabilities |
| centre / inner field | [B, 256, 256, 4] centre probabilities | [B, 256, 256, 1] normalised inner distance | Instance separation signal |
| offset | [B, 256, 256, 2] | Removed | Pixel-to-centre vector field in V6.2 |
| boundary | [B, 512, 512, 1] | [B, 512, 512, 1] | Boundary cue for conservative partitioning |

The V7 model builder validates the exact output-name and output-shape set. This is a strong protection against silent head-order errors: the target builder, loss compiler, output unpacker, decoder, and tests all expect semantic, inner_distance, and boundary.

V7 has 6,343,486 parameters, compared with 6,417,683 for V6.2: 74,197 fewer parameters, or about 1.16% smaller. The change is therefore not an attempt to win by material capacity growth. It replaces the instance representation and decoder while preserving most of the backbone capacity.

### Shared feature extractor

The retained backbone is technically appropriate for the task:

- RGB input is joined with a fixed Sobel edge representation.
- The encoder uses C3k2-style stages from P1/2 through P5/32.
- The main stage widths are 40, 72, 144, 256, and 384 channels.
- A spatial pyramid pooling fast block and channel attention provide high-level context.
- Two weighted bidirectional feature-pyramid passes fuse P3, P4, and P5 at a 160-channel pyramid width.
- Separate atrous-context branches feed semantic and instance-oriented decoding.

This is a reasonable multi-scale design for a dataset containing both small circular pads and much larger rectangular structures. The retained BiFPN-style fusion has an established rationale: repeated weighted bidirectional fusion was introduced to improve cross-scale feature exchange in EfficientDet, although that paper does not establish a performance result for this PCB task specifically. [EfficientDet](https://arxiv.org/abs/1911.09070)

### The V7 inner-distance target

The crucial V7 change is implemented by the inner-distance target builder:

1. Semantic and instance identifiers are nearest-neighbour resized to the 256 by 256 instance grid.
2. Background, frame pixels, and four-neighbour instance transitions are marked as boundaries.
3. A Euclidean distance transform is calculated within each instance interior.
4. Each instance is independently normalised by its own maximum distance.
5. The target supplies both the distance value and a foreground-validity mask.

This per-instance normalisation is important. A large rectangle and a tiny pad both retain an interior peak near one, so a shared core threshold does not automatically eliminate small instances merely because their absolute radius is smaller. The special treatment of one-pixel-wide degenerate instances also prevents an all-zero target for a valid tiny object.

The V7 loss is a masked Huber loss with delta 0.10, twice the weight at the rim, and per-image normalisation over valid weighted pixels. Empty-foreground images return zero distance loss rather than creating an invalid denominator. This is substantially better aligned with the target than treating the field as an unmasked image-regression problem.

### Loss-budget calibration

V7 compiles three losses:

| Loss | V6.2 weight | V7 weight | Interpretation |
|---|---:|---:|---|
| semantic | 1.0 | 1.0 | Primary class-mask supervision |
| centre | 0.25 | Removed | V6.2 centre heatmap |
| offset | 1.0 | Removed | V6.2 centre-vector supervision |
| inner distance | N/A | 10.0 | V7 instance-geometry supervision |
| boundary | 0.3 | 0.3 | Boundary refinement |

The V7 change record explains that a raw inner-distance loss near 0.005 would otherwise have negligible influence. With a multiplier of 10, it contributes approximately 0.05 to the mid-training loss budget, comparable to the combined V6.2 geometry contribution. This is a defensible calibration hypothesis, and the current loss contract test checks that the geometry share stays within an expected range. It is still a hypothesis until actual V7 training curves show that the semantic, distance, and boundary heads all learn at healthy rates.

## Decoder inspection

### V7 decoding path

V7 decodes each foreground class separately:

1. It applies semantic argmax and a semantic-confidence gate, currently 0.30.
2. It upsamples the 256-grid inner-distance field to the semantic grid.
3. It finds connected semantic foreground blobs.
4. Within each blob, it thresholds the predicted field relative to that blob's own maximum, currently 0.50.
5. It closes one-pixel core gaps, removes cores smaller than eight pixels, and retains a safe fallback if a valid blob has no usable core.
6. It expands surviving cores inside the original semantic blob using a distance-transform label assignment.
7. It applies conservative boundary-based partitioning where a boundary core is sufficiently credible.
8. It computes a per-component self-consistency score: agreement between the predicted inner-distance field and the distance field implied by the decoded component, multiplied by mean semantic confidence.

This is a stronger formulation than global-centre NMS plus offset clustering for dense, non-convex, or size-variable components. In particular, the core threshold is relative to the individual semantic component, which directly addresses the small-object failure mode that a fixed global heatmap threshold can create.

The strategy has meaningful literature precedent. Deep Watershed Transform uses a learned energy representation whose thresholded basins yield instances, while also noting the familiar failure mode of spurious partitions caused by noisy energy geometry. That supports V7's emphasis on per-blob safeguards, minimum core area, and degraded-field tests; it does not validate V7's accuracy by itself. [Deep Watershed Transform for Instance Segmentation](https://arxiv.org/abs/1611.08303)

### Decoder improvements that are present

The local V7 change report and current code agree that the following improvements are implemented:

| Improvement | Why it matters |
|---|---|
| Per-blob relative core threshold | Retains small objects beside large objects more reliably than a global field threshold |
| Per-blob expansion constraint | Prevents an instance seed from leaking through unrelated semantic regions |
| Conservative boundary partition | Uses boundary information without forcing a split from weak evidence |
| Per-component field-consistency score | Uses the complete interior geometry rather than only a peak value |
| Minimum-core and fallback handling | Reduces disappearance of valid small components |
| Independent V7 TTA module | Averages the three V7 heads rather than relying on the obsolete V6 output schema |

The prior change report also includes synthetic decoder ablations. Its reported score increases from 0.9905 to 0.9950 in an oracle setting, 0.9934 to 0.9950 under mild degradation, 0.9896 to 0.9923 under moderate degradation, and 0.9807 to 0.9894 under harsh degradation. These are useful regression indicators for the decoder, but they use oracle or deliberately controlled fields and are not V7 model-quality metrics. They must not be reported as validation or test mAP/F1.

### Tiled inference: implemented component, incomplete product path

The V7 source includes a thoughtful native-resolution tiling component:

- Tile size: 512 pixels.
- Overlap: 192 pixels.
- Effective stride: 320 pixels.
- Blend window: two-dimensional cosine taper with a protected blend margin.
- Head fields are blended first; decoding occurs once after blending.

The tile grid, blending ranges, output shapes, probability sums, and simple seam behaviour are covered by test_tier3_tiling. That is a positive implementation-level result.

However, the public prediction function letterboxes the original image to 512 by 512 and calls the model directly. It does not call predict_fields, which is the function that dispatches to tiled prediction for images larger than 512. Therefore the current public image-prediction route cannot exercise the tile system at native resolution. The change report correctly shows that dispatch exists inside predict_fields; it does not demonstrate end-to-end use through predict_one_image, predict_folder.py, or predict_pictures.py.

## Baseline evidence: trained V6.2 scratch model

### Artifact integrity and selection

The inspected trained baseline is /home/u117134c/Models/Model_v6_2_scratch_rtx. It contains initial random weights, a last checkpoint, and several selected checkpoint copies. The authoritative best-model file is byte-identical to the selected best-semantic file:

| Artifact | SHA-256 | Observation |
|---|---|---|
| best_semantic_model_v6_2.keras | e614ea930d0f3da64f2d90ec8088b33b0e54baa335ced2ef7efdd3504ffa5283 | Selected full-validation checkpoint |
| best_model_v6_2_instance.keras | e614ea930d0f3da64f2d90ec8088b33b0e54baa335ced2ef7efdd3504ffa5283 | Authoritative copy; byte-identical |
| best_instance_model_v6_2.keras | c0620f4d9e06d5b9dfe6d81caeb1c524352725e27909d63c6ba25fb797a7fcaa | Alternate checkpoint |
| last_model_v6_2.keras | 980a71aa850932d644ced224bb36f3b72ed2de098a4e1820e9529ac9479f5f56 | Final-epoch checkpoint |

The final selection record identifies best_semantic_model_v6_2.keras as the source selected by highest complete-validation instance F1 across 1,048 validation images. Its recorded model size is 6,417,683 parameters, matching the V6.2 source contract.

### Training configuration and convergence

The V6.2 run was a scratch run, not a transfer run. It used TensorFlow 2.21, Keras 3.15.1, mixed float16, batch size 2 with gradient accumulation 2 (effective batch size 4), AdamW with EMA 0.999, gradient clipping 5, six warm-up epochs, a 3e-4 to 1e-6 cosine schedule, and 120 scheduled epochs on an RTX PRO 1000 Blackwell Laptop GPU.

The CSV contains epoch indices 0 through 119. At the final logged epoch, train loss was 0.132368, validation loss was 0.201441, train foreground mIoU was 0.931890, validation foreground mIoU was 0.908343, and subset instance mAP50-95 was 0.830139 at learning rate 1e-6. The training report was generated before the final log entry and calls the run 119 of 120; the CSV is the more complete epoch-count record.

The loss and metric trajectory is consistent with convergence rather than a late collapse. It does not, by itself, prove ideal generalisation: the held-out test distribution is notably denser and smaller-object-heavy than training data.

### V6.2 result reference

The following values are baseline evidence only; they are not V7 results.

| Evaluation record | Instance F1 | Precision | Recall | mAP50 | mAP50-95 | Foreground mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Full validation selection record | 0.94970 | 0.92522 | 0.97551 | 0.97476 | 0.82979 | 0.90847 |
| Later full-validation performance summary | 0.96790 | 0.96964 | 0.96617 | 0.97920 | 0.84497 | 0.91254 |
| Full 432-image test evaluation | 0.97527 | 0.96961 | 0.98100 | 0.99021 | 0.77110 | 0.86296 |

The differing validation figures come from different recorded evaluation artifacts and should not be silently mixed. For a fair V7 comparison, reproduce one fixed protocol, one checkpoint-selection rule, one decoder configuration, and one clean test set.

The test evaluation counted 33,472 target instances and 33,865 predicted instances: 32,836 true positives, 1,029 false positives, and 636 false negatives. The loose-overlap result is strong, but the gap from mAP50 0.99021 to mAP50-95 0.77110 shows that high-IoU boundary/partition accuracy is substantially harder.

Per-class test mAP50-95 is especially informative:

| Class | Test mAP50-95 | Interpretation |
|---|---:|---|
| Rectangle | 0.82256 | Strong but still has tight-boundary room |
| Rectangle_concave | 0.86975 | High value but only 108 test instances; uncertainty is large |
| circle | 0.59024 | Primary structural weakness; very high loose-overlap result but weak tight-IoU result |
| circle_full | 0.80187 | Stronger than circle, still below loose-overlap behaviour |

The circle class has mAP50 0.9869 but mAP75 0.6132 and mAP95 0.0150 in the recorded test evaluation. This makes V7's boundary-aware, shape-aware interior-field representation a sensible hypothesis. It remains a hypothesis until V7 is trained and tested under the same protocol.

### Why V6.2 is a baseline, not a V7 initialisation

V6.2 uses centre and offset heads that are absent from V7. The V7 public launcher deliberately exposes scratch mode rather than V6.2-to-V7 transfer. A V6.2 checkpoint can therefore serve as a performance, speed, and data-protocol baseline, but it is not a structurally compatible V7 head initialisation in the current workflow.

## V6.3 and lineage assessment

The local V6.3 source snapshot is useful as a design reference but not as an empirical baseline. It has no supplied trained result directory or final evaluation artifact, and it is an untracked local file rather than a repository-identified release point.

Static comparison with V6.2 shows 218 inserted and 1,138 deleted lines. V6.3 retains the V6-style semantic, centre, offset, and boundary heads and their canonical decoder. It also simplifies several data/augmentation controls relative to V6.2, including the array-root handling, zoom configuration, copy-paste augmentation, and some label-overlap logic. It retains dihedral augmentation and the instance-F1 checkpointing mechanism.

The V7 change record describes V7 as based on the V6.4 state, while the V7 README describes a V6.2 fork lineage. These statements can be reconciled as follows: V7 is in the V6.2 family, with V6.4 as the direct intermediate implementation state. V6.3 should not be assigned a performance position without a corresponding trained artifact.

## Dataset and evaluation risk analysis

### Split shift

The dataset audit records 6,160 images: 4,680 train, 1,048 validation, and 432 test. The test set is substantially harder in object density and object scale:

| Indicator | Train | Validation | Test |
|---|---:|---:|---:|
| Objects per image | 32.4 | 46.2 | 77.5 |
| Rectangle median diameter | 35.7 | 27.7 | 21.9 |
| circle median diameter | 33.8 | 33.3 | 16.1 |
| circle_full median diameter | 21.0 | 17.5 | 16.0 |

This shift is directly relevant to V7. Small/dense components are the cases most exposed to core-selection, instance splitting, overlap resolution, and downsampled 256-grid geometry. A validation result alone may overestimate or mischaracterise gains on the true test distribution.

Training labels are heavily dominated by circle_full and Rectangle instances (76,612 and 57,857 respectively), followed by circle (12,000) and only 680 Rectangle_concave instances. The concave class's high test score should therefore be interpreted with care because it has low support, and its future V7 quality needs class-aware confidence intervals rather than a single aggregate metric.

### Split contamination and protocol mismatch

The audit found 44 byte-identical images shared between validation and test, approximately 10% of the test set. It also records label-overlap issues in 9.3% of training images, 8.0% of validation images, and 19.4% of test images.

The existing threshold-fit file was fitted on V6-style validation data and itself notes the 44 validation/test duplicates. Any test result using thresholds selected from a validation set containing exact test duplicates is partially contaminated. This does not invalidate the trained V6.2 model, but it does lower confidence in fine-grained deployment-threshold claims.

There is an additional comparability issue: the V6.2 run configuration refers to /home/u117134c/data/pcb_v62_arrays, while the current V7 active-run configuration points to /home/u117134c/data/pcb_v62_arrays_repaired. A future V7-versus-V6 comparison must either evaluate both models against the same repaired arrays and protocol or state explicitly that the comparison is historical rather than controlled.

## Operational findings and risks

### P0: public prediction fails because it references a deleted head

The highest-severity source issue is in model_v7.py's predict_one_image path. The function correctly unpacks V7's three outputs into semantic probabilities, small inner-distance field, and boundary probability. Later in the same function it tries to resize and use centre_probabilities_small and centres_original. No centre_probabilities_small value is defined in that V7 function because the centre head was removed.

This will raise a NameError during real prediction. It is not merely unused dead code:

- model_v7.py's normal prediction convenience path reaches predict_one_image.
- predict_folder.py delegates to the model prediction path.
- predict_pictures.py delegates to the same prediction path.

Impact: a trained V7 checkpoint would not be usable through the intended public inference interface until this is corrected and tested.

### P0: threshold fitting is still a V6 four-head program

Model_v7/fit_thresholds.py still unpacks four outputs as semantic, centre, offset, and boundary, and calls decode_instances with V6-style arguments. V7 output_probabilities returns exactly semantic, inner_distance, and boundary; V7 decode_instances accepts the corresponding three-field contract.

The program will fail immediately when it attempts to unpack the model output. More importantly, its parameter grid is conceptually stale: V7 needs calibration of the semantic gate, relative inner-distance core threshold, core-area rule, instance-area rule, boundary partition threshold, and final deployment score, not centre confidence and centre-NMS radius.

Impact: there is no valid way in the current public scripts to produce a calibrated V7 deployment profile.

### P1: tiled inference is not connected to end-user prediction

The tiling component passes its own tests, but public prediction letterboxes every original image to 512 and calls the model directly. It bypasses predict_fields, so larger native-resolution images do not use the intended tiles, overlap, or blended fields.

Impact: the codebase cannot currently substantiate its native-resolution tiled-inference claim through normal user-facing prediction.

### P1: benchmark mode targets a missing script

train_v7.py and the command help refer to train_rtx.py for benchmark execution, but there is no Model_v7/train_rtx.py file. The scratch launcher is separate and can be invoked, but the benchmark preflight/subprocess route cannot run as written.

Impact: the intended performance/memory benchmark should not be treated as available until its launcher target is restored or redirected.

### P1: deployment settings are inherited from V6 semantics

deployment.py imports the legacy threshold profile and still applies V6 centre-confidence and NMS concepts. Those controls are inert or meaningless for V7's distance-field decoder. It also uses V6-derived area/boundary values and a default minimum confidence of 0.50 without V7 field-consistency calibration.

Impact: even if inference did not fail, deployment output would not be evidence-based for V7.

### P2: source defaults can target stale locations or run modes

The module-level defaults in model_v7.py still include an older V5-augmentation output directory, a V5-transfer inference model location, and transfer as a default run mode. env.sh and train_v7.py override these for the intended scratch run, but direct module use can select stale paths.

Impact: this is a reproducibility and operator-error risk rather than an immediate training-algorithm error.

### P2: dashboard and documentation drift

dashboard.py still visualises centre_loss and offset_loss/weights instead of inner_distance loss. It will conceal the critical V7 geometry learning signal.

Several README statements are stale in the other direction: they describe configuration recording, V5 validator behaviour, per-blob handling, and tiled dispatch as unresolved even though the current source and local change record show them as implemented. The README is correct that V7 is untrained. Its other implementation-status claims should not be used as the authoritative current-state description.

The documented command ./run.sh predict-folder is not exposed by run.sh, although predict_folder.py exists. Watchdog comments also retain outdated session/script names. These are P2 workflow/documentation issues, but they raise the chance of incorrect operation after the P0 fixes.

## Positive implementation findings

The inspection found several areas where the current V7 code is stronger than its documentation suggests:

- The three-head output schema is explicitly validated.
- The target builder correctly performs per-instance distance normalisation and preserves tiny degenerate objects.
- The distance loss masks invalid/background regions and weights boundaries without invalid empty-image behaviour.
- The decoder operates per semantic blob rather than globally, reducing cross-object leakage.
- Instance ranking is based on whole-component field agreement rather than a peak-only score.
- The current configuration writer records V7's inner-distance head and loss weighting; the README note claiming otherwise is stale.
- V7 evaluation imports the V7-specific TTA path when TTA is enabled; generic V6 TTA code is not incorrectly used by that path.
- The five V7 contract tests pass independently in the inspected snapshot.

These are meaningful readiness signals for development. They are not substitutes for a checkpoint load/predict smoke test, a real validation sweep, or an unbiased test evaluation.

## Recommended gates before a full V7 campaign

The following are recommendations only; no changes were made as part of this inspection.

| Priority | Gate | Passing condition |
|---|---|---|
| 1 | Repair P0 prediction interface | A randomly initialised and a saved V7 model can execute predict_one_image, predict_folder, and predict_pictures without centre/offset references |
| 2 | Repair P0 calibration interface | A V7 validation run can fit and save V7-native decoder/deployment thresholds without output-schema errors |
| 3 | Complete native-resolution route | A normal public image prediction over 512 pixels demonstrably invokes tiled field blending and decodes once after blending |
| 4 | Restore benchmark path | The advertised benchmark command resolves to an existing script and records memory, time, and output integrity |
| 5 | Add V7 dashboard fields | Inner-distance loss, its weighting, and decoder-related metrics are visible during training |
| 6 | Run a short smoke campaign | Verify loss decreases, all three heads remain finite, checkpoints save/load, and validation decode completes |
| 7 | Run the controlled full campaign | Use an explicit source hash, repaired-array identity, fixed seed(s), and reproducible configuration artifact |
| 8 | Perform an unbiased comparison | Re-evaluate V6.2 and V7 with the same arrays, split exclusions, threshold-selection procedure, runtime hardware, and metrics |

For the V7 threshold sweep, the primary candidates should be the semantic confidence gate, relative inner-distance core threshold, minimum core area, minimum instance area, boundary-partition threshold, and final field-consistency confidence. Centre confidence and centre-NMS radius should be removed from the V7 calibration vocabulary.

The report should record at minimum mAP50, mAP75, mAP50-95, instance F1, precision, recall, foreground mIoU, per-class metrics, count error, small-object bins, latency, peak memory, and seed-to-seed variability. The V6.2 circle tight-IoU weakness means mAP75/mAP95 and small-circle strata are more diagnostic than a single aggregate F1.

To remove known evaluation bias, the 44 exact validation/test duplicates should be excluded from the final untouched test comparison, or an alternative clean test set should be declared before thresholds are fitted. Any historical V6 score that retained them should be labelled accordingly.

## Research context and interpretation limits

V7's direction is consistent with established instance-segmentation research:

- Learned watershed-energy approaches show that an interior geometric field can be converted to instances through thresholded basins, but also motivate strong protection against spurious partitions from imperfect fields. [Deep Watershed Transform for Instance Segmentation](https://arxiv.org/abs/1611.08303)
- Weighted, repeated bidirectional feature-pyramid fusion has an established multi-scale rationale, especially where objects appear at different sizes. [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)
- Cellpose is a useful example of a domain-specific instance-segmentation system using flow-like geometry and tiled/augmented inference ideas, but its microscopy evidence cannot be transferred directly to PCB imagery. [Cellpose: a generalist algorithm for cellular segmentation](https://www.nature.com/articles/s41592-020-01018-x)

These references establish methodological plausibility, not expected V7 mAP. The only credible answer to whether V7 improves the PCB system will be a controlled trained-model comparison after the P0/P1 pipeline issues are closed.

## Evidence inventory

| Evidence | Location | How it was used |
|---|---|---|
| V7 model, targets, losses, decoder, defaults | Model_v7/model_v7.py | Static source and head-contract inspection |
| V7 launcher/configuration | Model_v7/train_v7.py, env.sh, run.sh, active_run.env | Training and operational-path review |
| V7 tests | Model_v7/run_v7_tests.sh and test files | Independently executed contract suite |
| V7 change record | Model_v7/Change/CHANGE_REPORT.md | Cross-check for previously recorded synthetic decoder evidence |
| V7 threshold record | Model_v7/results/threshold_fit.json | Legacy-threshold incompatibility review |
| V7 deployment/dashboard/prediction helpers | deployment.py, dashboard.py, fit_thresholds.py, predict_folder.py, predict_pictures.py | Public-path and reporting review |
| Dataset audit | Model_v7/memory/dataset-split-data-audit.md | Split-shift and contamination analysis |
| V6.2/V6.3/V6.4 source | model-v6-2 | Architectural lineage comparison |
| Trained V6.2 artifacts | /home/u117134c/Models/Model_v6_2_scratch_rtx | Baseline integrity, configuration, training, and evaluation review |

## Final conclusion

V7 is not a minor V6.2 revision. It changes the instance representation in a way that is technically aligned with the principal observed weakness of the V6.2 baseline: precise separation and boundary fidelity for small/dense shapes, particularly circles. The target construction, loss, blob-local decoder, per-component scoring, and component tests are credible engineering improvements.

At the same time, V7 remains an untrained research candidate rather than an operational model. The strongest conclusion currently justified is that its internal contracts pass and its design merits controlled training. The strongest conclusion not justified is that it is accurate, faster, more robust, or better than V6.2.

Do not treat V7 as ready for production inference or deployment until the public prediction and V7-native threshold-fitting paths are repaired, tiled inference is connected end-to-end, and a clean, reproducible V7 evaluation demonstrates improvement against V6.2 on the same data protocol.
