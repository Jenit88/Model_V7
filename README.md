# Model V7: finding shapes on PCB images

Model V7 looks at a photo of a printed circuit board (PCB) and finds four kinds of shapes on it. For every shape it finds, it gives you:

- the **class**: which of the four kinds it is
- the **outline**: the exact pixels the shape covers, not just a box around it
- a **confidence score** from 0 to 1

The model was trained from scratch and is ready to use. Download the trained model file from the [Releases page](https://github.com/Jenit88/Model_V7/releases).

![Model V7 result on a test image](results/examples/sample_val_000017.png)

*Model V7 on a test image. Every shape it found is outlined and labelled with a number, its class and its confidence.*

## Contents

1. [What the model finds](#what-the-model-finds)
2. [Results](#results)
3. [How it works](#how-it-works)
4. [How it was trained](#how-it-was-trained)
5. [Use the trained model](#use-the-trained-model)
6. [Check the scores yourself](#check-the-scores-yourself)
7. [Train your own model](#train-your-own-model)
8. [Run the tests](#run-the-tests)
9. [Limitations and things to know](#limitations-and-things-to-know)
10. [Files in this repository](#files-in-this-repository)

## What the model finds

| class | shape | shapes in the test set |
|---|---|---:|
| `Rectangle` | rectangular pad | 15,156 |
| `Rectangle_concave` | rectangle with a hollow or cut-out part | 108 |
| `circle` | ring: a circle with a hole in the middle | 3,272 |
| `circle_full` | solid circle | 14,952 |

## Results

The model was scored on pictures it never trained on:

- **validation set**, 1,048 images: used to pick the best version of the model
- **test set**, 432 images: kept aside and used only at the very end

| data | images | shapes | mask mAP50-95 | mAP50 | mAP75 | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 1,048 | 48,388 | **0.8515** | 0.9874 | 0.9712 | 0.9598 | 0.9839 | **0.9717** |
| test | 432 | 33,488 | **0.7554** | 0.9731 | 0.8470 | 0.9577 | 0.9810 | **0.9692** |

On the test set, per class:

| class | shapes | mask mAP50-95 | precision | recall |
|---|---:|---:|---:|---:|
| Rectangle | 15,156 | 0.8300 | 0.9421 | 0.9817 |
| Rectangle_concave | 108 | 0.8235 | 0.5025 | 0.9167 |
| circle | 3,272 | 0.5649 | 0.9730 | 0.9795 |
| circle_full | 14,952 | 0.8032 | 0.9767 | 0.9811 |

**What the numbers mean.** All scores go from 0 to 1, and higher is better.

- **IoU (overlap):** how much a predicted outline and the real outline cover the same pixels. 1 means identical.
- **Precision:** of the shapes the model found, the share that are real.
- **Recall:** of the real shapes, the share the model found.
- **F1:** one number that balances precision and recall.
- Precision, recall and F1 count a found shape as correct when its IoU with a real shape is at least 0.50.
- **mask mAP50** and **mAP75:** average precision when a match needs an IoU of at least 0.50 or 0.75.
- **mask mAP50-95:** the same, averaged over ten levels from 0.50 to 0.95. It rewards outlines that fit exactly.

**In short:** the model finds almost every shape (recall 0.98), and nearly everything it reports is real (precision 0.96). What is left to improve is mostly how exactly the outlines fit, plus the weak spots listed under [Limitations](#limitations-and-things-to-know).

Full tables, training logs, evaluation reports and more example pictures are in [`results/`](results/README.md).

## How it works

**1. Resize.** The picture is scaled to 512 × 512 pixels without stretching. Grey bars fill any empty space.

**2. Predict three maps.** One neural network (6.3 million parameters) reads the picture and outputs three maps:

| map | size | what it gives for each pixel |
|---|---|---|
| class map | 512 × 512 | background or one of the four classes, with a probability |
| edge map | 512 × 512 | how likely the pixel is on the edge of a shape |
| inner-distance map | 256 × 256 | 0 at the edge of a shape, rising to 1 in its middle |

The inner-distance map is the main idea. Every shape, big or small, looks the same in it: low at the edge, high in the middle. So each shape's middle is easy to find, even when two shapes touch.

**3. Turn the maps into shapes.** For each class:

1. Keep the pixels the network is at least 30% sure about. Pixels that touch form a blob.
2. In each blob, the high part of the inner-distance map (at least half of the blob's highest value) marks the middle of each shape. Each middle becomes one shape.
3. Every other pixel in the blob joins its nearest middle.
4. The edge map separates shapes that are still stuck together.
5. Very small pieces (under 13 pixels) are thrown away.

**4. Score each shape.** The confidence score checks whether the predicted inner-distance map fits the shape's own outline, then multiplies that by how sure the network is about the class. A clean, whole shape scores high. A broken piece, or two shapes merged into one, scores lower.

**5. Scale back.** The outlines are mapped back to the size of the original picture.

**Inside the network:** the first layers look at both the colours and the edges of the picture (a fixed Sobel edge filter). An encoder then shrinks the picture step by step to learn bigger patterns. A feature pyramid (BiFPN) mixes fine and coarse detail. Two decoders build the maps: one makes the class map and the edge map, the other makes the inner-distance map.

## How it was trained

- **Data:** 4,680 training images (151,724 shapes), 1,048 validation images (48,388 shapes) and 432 test images (33,488 shapes). The labels are polygons in YOLO format. Where two label polygons overlapped, the overlap was cleaned up before training so every shape keeps its own pixels.
- **From scratch:** the network started with random weights. No pretrained weights were used.
- **Length:** 120 epochs (full passes over the training images), 11–14 September 2026, on one laptop GPU (NVIDIA RTX PRO 1000, 8 GB). One epoch takes about 34 minutes.
- **Augmentation:** each epoch shows the training images changed in different ways: rotations, flips, small zooms in and out, and changes to brightness, contrast, colour, blur and noise. This stops the model from relying on one exact view.
- **Loss:** one loss for each of the three maps. The inner-distance loss pays extra attention to pixels near the edges of shapes.
- **Settings:** AdamW optimiser. Learning rate 0.0003 with a 6-epoch warm-up, then lowered slowly to 0.000001 (cosine schedule). 2 images per step, with gradients added up over 2 steps (works like 4 images per step). Mixed precision (float16) to save GPU memory. A running average of the weights (EMA, 0.999).
- **Picking the best version:** from epoch 25, every 5 epochs the model was scored on 256 validation images and the best version so far was saved. At the end, three saved versions were compared on all 1,048 validation images. The version from **epoch 105** won; it is the released model.

## Use the trained model

**You need:** Linux, or Windows with WSL2 (Ubuntu); Python 3.11; an NVIDIA GPU (strongly recommended).

**1. Get the code and install the packages.**

```bash
git clone https://github.com/Jenit88/Model_V7.git
cd Model_V7
python3.11 -m venv ~/envs/pcb62
~/envs/pcb62/bin/pip install "tensorflow[and-cuda]==2.21.0" "keras==3.15.1" "numpy==2.4.6" "opencv-python-headless==5.0.0.93"
```

`env.sh` looks for the Python environment in `~/envs/pcb62`. If you made it somewhere else, run `export PCB_ENV=/your/env/folder` first.

**2. Download the trained model** (`best_model_v7_instance.keras`, 231 MB) into `~/Models/Model_v7_scratch_rtx/`:

```bash
mkdir -p ~/Models/Model_v7_scratch_rtx
gh release download v1.0 --repo Jenit88/Model_V7 --dir ~/Models/Model_v7_scratch_rtx
```

You can also download it from the [Releases page](https://github.com/Jenit88/Model_V7/releases) in a browser. To keep it in another folder, run `export PCB_MODEL_OUTPUT_DIR=/that/folder`.

**3. Check that TensorFlow can see the GPU.**

```bash
source env.sh
$PCB_PYTHON -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If this prints `[]`, TensorFlow cannot see the GPU and will not use it.

**4. Find shapes in your pictures.**

To save a result picture for every image in a folder (needs a GPU):

```bash
source env.sh
$PCB_PYTHON predict_pictures.py /path/to/images /path/to/output
```

Each image gets a result picture with the same name in `/path/to/output`. If the run stops, start it again: finished images are skipped.

To try the model on a random sample of a folder (also works without a GPU, just slower):

```bash
PCB_SOURCE=/path/to/images PCB_PREDICT_OUT=/path/to/output PCB_SAMPLE=40 ./run.sh predict-folder
```

This picks 40 random images. It saves the result pictures in `/path/to/output/overlays/` and a `summary.json` with the number of shapes per class and how the confidence scores are spread. The full results for each image are in `~/Models/Model_v7_scratch_rtx/predictions/` (this folder is emptied at the start of every run):

| file | contents |
|---|---|
| `instances.png` | the picture with every shape outlined and labelled |
| `instances.json` | every shape: class, confidence, size and position |
| `instance_ids.npy` | for each pixel, the number of the shape it belongs to |
| `semantic_ids.npy`, `semantic_colour.png` | for each pixel, its class |
| `semantic_confidence.png`, `boundary.png`, `inner_distance.png` | the network's maps as heat maps |

Both scripts drop shapes with a confidence below **0.50**. To change that, put `PCB_MIN_CONFIDENCE=0.3` (or another value) in front of the command. To keep every shape, put `PCB_DEPLOYMENT_PROFILE=0` in front of it.

## Check the scores yourself

You need the labelled dataset, prepared as described in [Train your own model](#train-your-own-model) (steps 1 and 2). Then run:

```bash
./run.sh reevaluate
```

This scores the model on the validation and test sets, prints a table and saves the numbers in `results/reevaluation_reeval.json`. Full reports go to `~/Models/Model_v7_scratch_rtx/performance/`. Put `PCB_USE_TTA=1` in front of the command to average each prediction over 8 flipped and rotated copies of the image (slower).

## Train your own model

**1. Arrange the dataset like this:**

```text
Split_Data/
├── images/
│   ├── train/     pictures (.png, .jpg, ...)
│   ├── val/
│   └── test/
└── labels/
    ├── train/     one .txt file per picture, with the same name
    ├── val/
    └── test/
```

Each line in a label file is one shape in YOLO polygon format: the class number, then the polygon's x and y points, scaled from 0 to 1. For example:

```text
4 0.412 0.550 0.418 0.548 0.423 0.556 0.415 0.561
```

Class numbers: 1 = `Rectangle`, 2 = `Rectangle_concave`, 3 = `circle`, 4 = `circle_full`.

**2. Prepare the data.** Open `env.sh` and set `PCB_DATASET_ROOT` to your `Split_Data` folder. Then run:

```bash
./run.sh prepare
```

This converts the pictures and labels into arrays (about 10 GB) in the folder set by `PCB_ARRAY_DIR` in `env.sh`. Keep them on a fast Linux disk: reading them from a Windows drive (`/mnt/c`) is much slower.

**3. Train.** Use a new, empty output folder:

```bash
export PCB_MODEL_OUTPUT_DIR=~/Models/my_v7_run
./session.sh start scratch
```

Training runs in the background inside tmux (install `tmux` first). While it runs:

- `./session.sh attach` shows the training (press Ctrl-b, then d, to leave it running)
- `./session.sh status` shows a short summary
- `./run.sh report` prints the progress so far
- the live dashboard is at http://localhost:8088 and TensorBoard at http://localhost:6006

You can also run `./run.sh scratch` to train in the current terminal instead. 120 epochs take about 3 days on an 8 GB laptop GPU. When training ends, the best version is saved as `best_model_v7_instance.keras` in the output folder and scored on the validation and test sets.

## Run the tests

```bash
./run_v7_tests.sh    # six test suites on the CPU, no dataset needed
./run.sh selftest    # the model's built-in self-test
```

## Limitations and things to know

- **A single Rectangle among many round pads is often missed.** In the 20 test images with 1 `Rectangle` and 71 `circle_full`, the model missed the `Rectangle` in 10.
- **False `Rectangle_concave` detections.** On the test set, the model reported 98 wrong `Rectangle_concave` shapes against 108 real ones (precision 0.50). Most of them are on images that contain only 3 shapes.
- **Ring-shaped `circle` outlines are less exact.** `circle` has a mask mAP75 of 0.55, against 0.91 to 0.96 for the other classes. The inner-distance map is predicted at 256 × 256, where a thin ring is only 1 to 2 pixels wide.
- **Large pictures lose detail.** Every picture is shrunk to 512 × 512 first, so very small shapes on a large photo are harder to find. Code for predicting tile by tile at full resolution is included (`predict_fields` in `model_v7.py`), but the prediction scripts do not use it yet.
- **The decoding thresholds have not been tuned for this model.** Run `./run.sh fit-thresholds` to search for better values on the validation set. It saves them to `results/threshold_fit_v7.json`, and the prediction scripts then use them automatically.
- **The test set is not fully independent.** 44 test images are identical to validation images, and the validation set chose the best version.
- **Test pictures differ from training pictures.** In the test images, shapes are about 0.6× the size and 2.4× more tightly packed than in the training images.
- **`Rectangle_concave` is rare.** Only about 680 of the 151,724 training shapes belong to it.
- **Folder paths.** `env.sh` contains folder paths from the computer the model was trained on. Change them, or set the `PCB_*` variables, to match your computer.

## Files in this repository

| file | what it is |
|---|---|
| `model_v7.py` | the whole model: data preparation, network, training, prediction and evaluation |
| `train_v7.py` | training settings for an 8 GB GPU (used by `run.sh`) |
| `run.sh` | one command for every task: `selftest`, `prepare`, `scratch`, `report`, `predict-folder`, `reevaluate`, `fit-thresholds` and more |
| `env.sh` | environment setup, loaded by `run.sh` |
| `session.sh` | runs a long job in the background with tmux |
| `dashboard.py` | live training dashboard |
| `predict_pictures.py` | finds shapes in a folder of pictures and saves result pictures |
| `predict_folder.py` | the same on a random sample, plus a summary of the scores |
| `deployment.py` | settings for real boards (the 0.50 confidence cut-off) |
| `reevaluate.py` | scores the model on the validation and test sets |
| `fit_thresholds.py` | searches for better decoding thresholds |
| `tta.py`, `tta_v7.py` | averages predictions over flipped and rotated copies (test-time augmentation) |
| `run_v7_tests.sh`, `run_selftest_v7.py`, `test_*.py` | tests |
| `results/` | training outputs, evaluation reports, per-image results and example pictures |
| `TRAINING.md`, `change.md`, `change_2.md`, `inspection.md` | development notes, written before the final training run |
