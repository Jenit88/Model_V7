# Model_v7 — training readiness and launch

**Verified 2026-09-11 on this machine.** The model compiles, takes real gradient steps, decodes
validation, writes previews, and predicts end to end. It has never been trained.

---

## Readiness checks, and what each one proved

| check | result |
|---|---|
| GPU visible | RTX PRO 1000 Blackwell, 8,151 MiB, idle |
| Prepared arrays | `~/data/pcb_v62_arrays_repaired`, all 9 files present (train 3.7 GB) |
| Contract suites | **6/6 pass** (`./run_v7_tests.sh`) |
| Real training steps | **741 ms/step, peak 3.64 GiB, 28.9 min/epoch** |
| Data loader | 25.5 ms/image single-thread → **4.3 ms/image** with 6 workers |
| Loss balance at init | arithmetic reconstructs the logged total exactly (below) |
| Public prediction | `predict_one_image` runs on a real model and writes artifacts |
| Validation previews | one 2×4 panel sheet written per image |
| Supervision | `./v7_supervisor.sh status` reads the right run and target |

V6.2 for comparison: 752 ms/step, 3.70 GiB. V7 is marginally faster and leaner — one head
fewer, 6,343,486 parameters against 6,417,683.

### The loss balance, measured on the real model

From the preflight dump of the launched run:

| head | raw | × weight | contribution | share at init |
|---|---:|---:|---:|---:|
| semantic | 1.0832 | 1.00 | 1.0832 | 73.4% |
| **inner_distance** | 0.0222 | **10.00** | 0.2218 | **15.0%** |
| boundary | 0.5719 | 0.30 | 0.1716 | 11.6% |
| | | | **1.4766** | logged total 1.4766 |

This is the check that mattered. `INNER_DISTANCE_LOSS_WEIGHT` was inherited as 1.00 from the
offset head; at that value the only head that separates instances would hold **1.5%** of the
budget at initialisation. At 10.00 it holds 15.0% early and rises toward ~28% as the semantic
loss converges from 1.08 toward 0.075 — matching the 29.1% that centre+offset held in V6.2.

The offline estimate used to pick the weight predicted 0.024 for this loss at initialisation;
the real model gives 0.0222. The calibration measurement was sound.

---

## Launching

```bash
cd /mnt/c/Users/u117134/Desktop/dev/Model_v7

./run.sh selftest                 # ~1 min, no GPU needed
./session.sh start scratch        # the run — detached tmux session 'v7'
./watchdog.sh start               # restart on process death, inside WSL

# survive a host reboot — register AFTER the first launch, not before
powershell -ExecutionPolicy Bypass -File register_v7_watchdog.ps1
```

`env.sh` already defaults to `~/data/pcb_v62_arrays_repaired` and
`~/Models/Model_v7_scratch_rtx`, so no overrides are needed for the standard run.

> **Register the logon task after launching, not before.** `v7_supervisor.sh` is a *resume*
> tool: with no `fit_backup` it refuses rather than starting a run from scratch, which is
> correct but means it cannot perform the first launch.

### Watching it

```bash
./session.sh attach          # windows: work / dash / tb / gpu / dog
./session.sh status
./run.sh report
```

Dashboard <http://localhost:8088> · TensorBoard <http://localhost:6006>.

The dashboard now plots **inner-distance loss**. It was plotting centre and offset — two heads
V7 does not have — so it showed empty panels and hid the only curve that says whether instance
separation is being learned.

### Validation previews

`PREVIEW_IMAGE_COUNT = 20`, `PREVIEW_EVERY_N_EPOCHS = 10`. One 2×4 panel sheet per image:

```
<output>/validation_previews/epoch_0010_val_00285.png
```

Eight panels: image · ground-truth semantic · predicted semantic · error map ·
**target inner distance** · **predicted inner distance** · predicted boundary · decoded instances.

The two field panels are the ones to watch: they show directly whether the inner-distance head
is converging, which is the open question the loss weight leaves.

> The preview callback was unpacking four heads and raising on every preview epoch. Its wrapper
> catches exceptions and prints a warning, so a run would have produced **no previews at all**
> while looking healthy. `test_v7_predict.py` now asserts files are written.

---

## The first experiments, in order

### 1. The loss weight — short runs, not full ones

The weight is the one change with no end-to-end validation, and it cannot have any until a run
exists. The obvious sweep — weights 5/10/15 × several seeds — is **~25 days of GPU** at 2.8 days
a run. That budget does not exist here.

**Ask the same question in ~18 hours.** V6.2 reached mask mAP50-95 of 0.7284 by **epoch 10**, so
the signal separating a healthy head from a starved one is present early:

```bash
for w in 5 10 15; do
  PCB_EPOCHS=12 PCB_MODEL_OUTPUT_DIR=$HOME/Models/v7_w$w \
    ./session.sh start scratch          # edit INNER_DISTANCE_LOSS_WEIGHT between runs
done
```

Compare on epoch-10 instance mAP **and** on the inner-distance loss trajectory. Only the winner
needs a full run.

`session.sh` and `watchdog.sh` now forward `PCB_EPOCHS` and the other experiment knobs into the
tmux pane. They previously forwarded only the three path variables, so
`PCB_EPOCHS=12 ./session.sh start scratch` silently started a full 120-epoch run.

### 2. Mosaic — the measured distribution gap

`MOSAIC_PROBABILITY = 0.00` by default. The training distribution is measurably wrong for the
split it is scored on:

| | obj/img | Rectangle | circle | circle_full |
|---|---:|---:|---:|---:|
| train, augmented | 20.5 | 42.4 | 40.9 | 28.9 |
| **+ mosaic** | **93.0** | **19.0** | **20.6** | **12.2** |
| TEST | 78.0 | 21.9 | 16.0 | 15.9 |

Mosaic closes 81% of the gap, and zoom cannot: shrinking objects never adds any, so `obj/img`
held at 14.5 across every zoom-out setting tried. Cost is affordable — even p = 1.00 is
17 ms/image against a 370 ms/image step.

**Try `MOSAIC_PROBABILITY = 0.25` as an A/B against 0.00 on a short run before adopting it.**
It is off by default because the V6.2 Tier-1 run was an augmentation change argued from exactly
this reasoning, and it measurably cost mask precision. A mixture is likely better than a match:
validation sits at ~16 objects/image and test at ~78, so the model must handle both, and
p = 1.00 slightly overshoots test density.

### 3. Selecting checkpoints on clean validation

44 validation images are byte-identical to test images — **10.2% of the test split**,
recomputed here rather than taken on trust. Their indices are in
`results/validation_test_duplicates.json`; clean validation is 1,004 images.

Exclude them when selecting checkpoints, fitting thresholds, or quoting a test number. Selection
already uses `mask_map50_95` rather than F1, which is right — F1 at IoU 0.50 cannot tell a
barely-passing mask from an exact one.

### 4. Thresholds — after a checkpoint exists, not before

```bash
PCB_FIT_IMAGES=400 PCB_FIT_DRAWS=150 ./run.sh fit-thresholds
```

Searches V7's six controls: semantic gate, relative core threshold, minimum core area, minimum
instance area, boundary threshold, and the component-consistency floor. Writes
`results/threshold_fit_v7.json`.

The V6 fit is quarantined as `results/threshold_fit_v6_legacy.json` and `deployment.py` refuses
it: two of its five values set attributes V7's decoder never reads, and the other three were
fitted against a different grouping rule. Applying it would have succeeded silently.

**Per-class and size-stratified thresholds are deliberately not in the search.** Six dimensions
become roughly twenty-four split four ways, fitted on validation that is 10.2% test — and the
V6 fit is the cautionary case: it survived its held-out half and still scored *worse* on test
(0.7710 → 0.7694). Add those dimensions when a trained model shows a class or size band failing
for a reason a shared threshold cannot express.

---

## What is still not verified

1. **No V7 model has been trained.** Nothing here supports a claim about accuracy.
2. **The loss weight has no end-to-end validation** and is the most likely thing to need
   revisiting. Re-check the share at epochs 5, 20 and 50 against `training_log.csv`.
3. **Tiled inference is reachable and tested, but `predict_one_image` still letterboxes to 512
   before predicting**, so a full-resolution board does not route through `predict_fields` yet.
4. **Mosaic is measured on distribution, not on accuracy.** That is exactly the gap that made
   Tier-1 look justified.

## Stopping

```bash
./watchdog.sh stop     # first, or the supervisor restarts what you kill
./session.sh stop
powershell -ExecutionPolicy Bypass -File register_v7_watchdog.ps1 -Remove
```
