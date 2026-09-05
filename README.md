# CC1 Vision

**ML-Based Virtual Stain Detection & Cleaning Intelligence**

An academic computer-vision project that finds stains and dirty areas in a floor
image, classifies what kind of stain each one is, how severe it is, and what a
cleaning robot should do about it.

CC1 Vision is a **hybrid OpenCV + CNN system**. OpenCV *proposes* candidate
regions by comparing every part of the floor with its own local surroundings; a
convolutional neural network, trained locally with TensorFlow/Keras, then
*decides* which candidates are genuine stains and which class each belongs to.
The confidence shown in the dashboard is the network's own output probability.

Everything runs locally on the CPU. No physical robot, GPU, API key, cloud
service, paid API or pretrained-model download is required — the network is
trained from scratch on this machine, on data the project generates itself.

> **Accuracy statement.** The classifier achieves **96.4% test accuracy on the
> generated/synthetic test dataset**. This is *not* a measurement of real-world
> accuracy on photographs of real floors. See
> [Model performance](#model-performance) and
> [Academic Limitations](#academic-limitations).

---

## Table of contents

- [Project overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technologies used](#technologies-used)
- [Installation](#installation)
- [Running the application](#running-the-application)
- [How the OpenCV + CNN hybrid pipeline works](#how-the-opencv--cnn-hybrid-pipeline-works)
- [Stain classes](#stain-classes)
- [Dataset generation](#dataset-generation)
- [Model performance](#model-performance)
- [Retraining](#retraining)
- [Screenshots](#screenshots)
- [Project folder structure](#project-folder-structure)
- [Limitations](#limitations)
- [AI Assistance Disclosure](#ai-assistance-disclosure)
- [Academic Limitations](#academic-limitations)
- [Troubleshooting](#troubleshooting)

---

## Project overview

Commercial cleaning robots typically clean on a fixed schedule and a fixed
route. They cover clean floor at the same intensity as dirty floor, which wastes
water, battery and time, while heavily soiled spots may receive only a single
pass.

A robot that could *see* which parts of the floor are dirty, how dirty they are,
and what kind of soiling is present could prioritise its work. The vision and
decision layer of such a system can be developed and evaluated entirely in
software, which is what this project does.

Give it a photograph of a floor and it returns:

- outlined and boxed stain regions with a CNN class and confidence,
- a severity rating per region,
- the percentage of floor area that is dirty,
- a recommended cleaning action, tool, water level and pass count per stain,
- a priority-sorted robot task queue,
- a 2D floor map with a simulated cleaning route,
- downloadable annotated images plus JSON and CSV reports,
- a model-performance report for the classifier itself.

This is an academic concept inspired by autonomous cleaning systems. It is not
affiliated with, endorsed by or connected to any robot manufacturer, and it uses
no third-party trademarks or logos.

### Objectives

1. Detect stain-like regions in a floor image using classical computer vision.
2. Distinguish real soiling from floor structure such as grout lines, shadows
   and reflections.
3. Classify each region with a genuine trained model, not a hand-written rule
   set, and report the model's own confidence.
4. Quantify the affected floor area.
5. Convert detections into concrete cleaning actions with priorities.
6. Simulate how a robot would sequence those targets.
7. Present everything in a dashboard and export machine-readable reports.
8. Keep an honest fallback for machines where TensorFlow is unavailable.

---

## Features

| Feature | Status |
|---|---|
| Upload JPG / JPEG / PNG / WEBP | Working |
| OpenCV anomaly-based candidate generation | Working |
| CNN stain classification (TensorFlow/Keras, trained locally) | Working |
| CNN veto of non-stain candidates (grout, glare, shadow, border) | Working |
| Confidence taken from the CNN output probability | Working |
| Adjustable CNN acceptance threshold | Working |
| Synthetic labelled dataset generator (6 classes, 3 splits) | Working |
| Model performance tab (accuracy, precision, recall, F1, confusion matrix) | Working |
| Fragment merging into one physical stain | Working |
| Contours and bounding boxes with labels | Working |
| Severity: Low / Medium / High | Working |
| Dirty floor coverage percentage | Working |
| Anomaly / detection mask view | Working |
| Sensitivity and minimum-area controls | Working |
| Grout, shadow and glare suppression | Working |
| Detection table with classifier provenance | Working |
| Cleaning recommendations per stain | Working |
| CC1 Cleaning Decision Engine (priority queue) | Working |
| 2D virtual floor map with simulated route | Working |
| Annotated image / JSON / CSV downloads | Working |
| Save reports to disk | Working |
| Synthetic sample floor generator | Working |
| Headless self-test (13 checks) | Working |
| Heuristic fallback when no model is available | Working |

---

## Architecture

```
                      ┌────────────────────────────────┐
   floor image ─────▶ │  utils.load_image_from_bytes   │  validation, decoding
                      └───────────────┬────────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │  detector: OpenCV stage        │  candidate proposal
                      │  • local background model      │
                      │  • colour/brightness/texture   │
                      │  • anomaly map + threshold     │
                      │  • morphology + contours       │
                      │  • merge stain fragments       │
                      │  • geometric FP filters:       │
                      │    thin lines, grout joints,   │
                      │    glare, shadows, tile noise  │
                      └───────────────┬────────────────┘
                                      ▼  candidate regions
                      ┌────────────────────────────────┐
                      │  ml_classifier.StainClassifier │  the decision maker
                      │  • square context crop 96x96   │
                      │  • CNN forward pass (batched)  │
                      │  • argmax == clean  -> REJECT  │
                      │  • p < threshold    -> REJECT  │
                      │  • else: class + probability   │
                      └───────────────┬────────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │  detector: border/corner rule  │
                      │  edge regions need p >= 0.85   │
                      │  severity from p and area      │
                      └───────────────┬────────────────┘
                                      ▼  List[Detection]
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐
│ cleaning_engine   │   │ detector.draw_detections │   │ utils.build_*_report│
│ actions, priority │   │ render_mask_view         │   │ JSON / CSV / table  │
└─────────┬─────────┘   └──────────────────────────┘   └─────────────────────┘
          ▼  List[CleaningPlan]
┌───────────────────────────┐
│ route_simulator.plan_route│  priority tiers + nearest neighbour
│ render_floor_map          │
└───────────────────────────┘
                                      ▼
                              app.py  (Streamlit UI)
```

Import direction is one-way, so there are no circular imports:

```
app.py ──▶ detector.py ──▶ ml_classifier.py ──▶ utils.py
   │  │                └──▶ utils.py
   │  ├───▶ ml_classifier.py ──▶ utils.py
   │  ├───▶ cleaning_engine.py ──▶ utils.py
   │  ├───▶ route_simulator.py ──▶ cleaning_engine.py, utils.py
   │  └───▶ sample_generator.py ──▶ utils.py
   │
train_classifier.py ──▶ dataset_generator.py ──▶ sample_generator.py, utils.py
```

`detector.py` imports `ml_classifier` lazily, inside the function that needs it,
so the detector still imports and runs on a machine with no TensorFlow.
`train_classifier.py` and `dataset_generator.py` are offline tools; the
dashboard never imports them.

---

## Technologies used

| Layer | Technology |
|---|---|
| Language | Python 3.10 – 3.12 (TensorFlow has no 3.13/3.14 wheels yet) |
| Interface | Streamlit 1.63 |
| Computer vision | OpenCV 4.14 (headless build) |
| Machine learning | TensorFlow 2.21 / Keras 3.15, CNN trained from scratch |
| Metrics | scikit-learn (precision, recall, F1, confusion matrix) |
| Numerics | NumPy 2.x |
| Tables and CSV | pandas 2.x |
| Image IO fallback | Pillow |

The five packages in `requirements.txt` are enough to run the dashboard in
heuristic fallback mode. `requirements-ml.txt` adds TensorFlow and scikit-learn,
which the CNN path needs.

---

## Installation

### Windows

Open **Command Prompt** in the project folder (the one containing `app.py`):

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-ml.txt
```

`requirements-ml.txt` includes `requirements.txt`, so this installs everything:
Streamlit, OpenCV, TensorFlow and scikit-learn.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-ml.txt
```

### A note on the Python version

TensorFlow publishes wheels for **Python 3.10 – 3.12**. On a newer interpreter
(3.13 or 3.14) `pip install tensorflow` fails with *"No matching distribution
found"*, and the dashboard falls back to heuristic labels. If that happens,
create the environment with a supported interpreter:

```bat
py -3.12 -m venv .venv
```

or fetch a standalone build with `uv`, which needs no admin rights:

```bat
pip install uv
uv python install 3.12
uv venv --python 3.12 .venv
```

### No training required after cloning

`models/stain_classifier.keras` is committed to this repository, so the CNN
works immediately after cloning. You only need `dataset_generator.py` and
`train_classifier.py` if you want to retrain — see [Retraining](#retraining).

---

## Running the application

```bat
python -m streamlit run app.py
```

The dashboard opens at **<http://localhost:8501>**.

`python -m streamlit run app.py` is preferred over `streamlit run app.py`
because it works even when the Scripts folder is not on your PATH, which is
common on Windows. Windows users can also double-click **`run.bat`**, which
creates the environment on first run and then starts the dashboard.

### Verifying the installation without a browser

```bat
python selftest.py
```

This runs the whole pipeline headlessly — model file, model loading, CNN
inference, probability validity, candidate detection, the single-stain
regression, cleaning engine, route simulator, report generation and bad-input
handling — and prints a pass/fail line for each. Exit code 0 means everything
passed.

```
==================================================================
 CC1 Vision - self test
==================================================================
[ OK ] Trained model present
[ OK ] Model loads
[ OK ] CNN inference and probabilities
[ OK ] Sample generation
[ OK ] OpenCV + CNN hybrid pipeline
[ OK ] Single central stain regression
[ OK ] Detection pipeline
[ OK ] Sensitivity control
[ OK ] Cleaning decision engine
[ OK ] Route simulation and rendering
[ OK ] Report generation
[ OK ] Edge cases and bad input
[ OK ] Many detected regions
==================================================================
 All 13 checks passed.
```

### Using the dashboard

1. Press **Load sample floor** to analyse the built-in synthetic floor, or
   upload your own photo.
2. Adjust **Detection Sensitivity** in the sidebar until the result looks right.
3. Read the results across the five tabs: Detection, Cleaning decisions, Floor
   map, Reports, Model performance.
4. Download the annotated image, JSON report and CSV report from the Reports
   tab.

Sidebar controls:

| Control | What it does |
|---|---|
| Detection Sensitivity | Lowers the anomaly threshold. Higher finds fainter marks and more candidates. |
| Minimum Detection Area | Ignores regions smaller than this many pixels. |
| Show Detection Mask | Toggles the anomaly heat map view. |
| Show Bounding Boxes / Contours / Confidence | Control what is drawn on the analysis image. |
| Minimum Confidence | Discards weak detections on the heuristic fallback path. |
| Analysis Resolution | Longest side used for processing. Lower is faster. |
| Suppress tile joints / shadows / reflections | The three geometric false-positive filters. |
| Use trained CNN classifier | Switches the CNN decision stage on or off. |
| Minimum CNN stain probability | Softmax floor for accepting a candidate as a stain (default 0.60, range 0.30 – 0.95). |
| Floor map settings | Zone width in metres, robot speed, route display. |

---

## How the OpenCV + CNN hybrid pipeline works

The division of labour is deliberate: **geometry localises, data classifies.**
OpenCV is good at finding *where* something differs from its surroundings, and
bad at deciding *whether that difference is dirt*. A trained network is the
other way round. Stages 1–13 are classical computer vision and produce
*candidate regions*; stages 14–16 are the CNN and produce the actual detections.

**1. Load and resize.** The image is validated and, if its longest side exceeds
the analysis resolution, downscaled with the aspect ratio preserved.

**2. Noise reduction.** A bilateral filter removes sensor noise while keeping
stain edges sharp.

**3. Colour spaces.** The image is converted to LAB (perceptual lightness plus
two colour axes), HSV (for saturation and hue) and grayscale (for texture).

**4. Local background model.** This is the core idea. A large-kernel median
filter estimates what each area of floor *should* look like. The median ignores
stains, because a stain is small compared with the kernel, so what remains is
the underlying floor colour and the lighting gradient. For speed the median runs
on a quarter-scale copy and is then upsampled.

**5. Compare each pixel with its local floor.** Five difference maps are built:

| Map | Meaning |
|---|---|
| darkness | how much darker than the surrounding floor (LAB L) |
| brightness | how much lighter than the surrounding floor (LAB L) |
| colour | distance from the local floor colour (LAB a/b) |
| saturation | extra colour intensity (HSV S) |
| texture | local standard deviation above the floor's own texture |

**6. Robust scaling.** Each map is divided by `median + 3 × MAD-sigma`, with an
absolute minimum floor value. The absolute floor matters: without it, a
perfectly clean floor would produce large scores simply because its own noise is
tiny.

**7. Fusion.** The five maps combine into one 0–1 anomaly score using a weighted
soft-OR, so a single strong cue is enough to flag a region.

**8. Grout suppression.** Tile joints are long, continuous and thin. Structures
that survive a long one-pixel-wide morphological opening but are destroyed by a
thick square opening are marked as joints and damped by 97% in the anomaly map
before thresholding. Joints are probed at four orientations (0°, 45°, 90°, 135°)
so a floor photographed at an angle is handled, and each direction is closed
first so a joint interrupted by a stain is still recovered as one line.

**9. Threshold.** The sensitivity slider maps to a threshold between 0.62 and
0.16. A coverage guard raises the threshold automatically if more than 55% of
the image would be flagged, so a heavily textured floor reports its worst areas
instead of everything.

**10. Morphology.** Opening removes specks. Closing uses a disc sized from
`merge_gap_ratio` (3% of the shorter side), which is the first stage of fragment
merging: a stain broken apart by a grout line or a highlight is rejoined here.

**11. Contours → candidate regions.** Contours become *candidates*, not
detections. The area gate here is deliberately loose — one eighth of
`min_area_px` — because a faint stain often arrives as several small pieces, and
discarding them now would destroy the fragments stage 12 exists to reassemble.

**12. Fragment merging.** Candidates whose bounding boxes are within
`merge_gap_ratio × min(height, width)` pixels of each other are grouped with a
union-find pass, rasterised together, and closed with a disc wide enough to
bridge that distance, so each group returns as **one** outline. The full
`min_area_px` threshold is applied afterwards, to the merged region. A mud patch
with its splashes therefore reports as a single stain, and `merged_from` in the
JSON report records how many pieces went into it.

**13. Geometric false-positive filters.** For each candidate the detector
measures mean darkness, brightness, colour, saturation, texture, hue (circular
mean), edge strength along the boundary, solidity, circularity, elongation and
rotated-rectangle fill, then rejects:

| Filter | Rule |
|---|---|
| Thin straight line | elongation ≥ 7 with a very thin short side, **or** elongation ≥ 4.5 with a thin short side that fills ≥ 55% of its own rotated bounding box |
| Grout joint | more than 50% of the region overlaps the detected joint mask |
| Glare | bright, colourless, smooth **and** soft-edged (grainy dust survives this) |
| Shadow | darker with almost no colour deviation, no extra texture, soft boundary |
| Large uniform shadow | ≥ 3% of the image, darkening, colourless, textureless, soft-edged |
| Tile texture noise | smaller than 0.035% of the image **and** weakly anomalous |

The texture rule deliberately combines size *with* weakness. Raising the minimum
area alone would hide false positives at the cost of missing real small stains;
requiring a region to be both tiny and faint does not.

**14. CNN classification.** Every surviving candidate is cropped as a square
`1.6 ×` the region's own size — the network needs surrounding floor to judge
whether the region differs from it — padded by reflection when the crop runs off
the image, resized to 96×96 and classified in a single batched forward pass.

- If the winning class is `clean`, the candidate is **rejected**. This is where
  the grout lines, reflections, shadow edges and speckle that survived the
  geometric filters are removed: by a trained model, not by another rule.
- If the winning stain class scores below `min_stain_probability` (0.60 by
  default, adjustable in the sidebar between 0.30 and 0.95), the candidate is
  rejected as too uncertain.
- Otherwise the region becomes a detection, labelled with the predicted class
  and carrying that class's softmax probability as its confidence.

**15. Image border and corner rule.** A region is treated as a border case when
at least 55% of it lies inside a 2%-wide band around the image edge, or when its
bounding box touches two perpendicular edges (a corner). Such a region is only
kept when the classifier scores it at 0.85 or above. Vignetting, the frame of
the photograph and whatever the camera clipped are therefore discarded, while a
real stain that runs off the edge of the picture still survives.

**16. Severity.** `0.55 × confidence + 0.45 × normalised area`, thresholded into
Low / Medium / High. With the CNN active, the confidence in that formula is the
network's probability.

### Where each displayed number comes from

| Value shown | Source |
|---|---|
| Stain type | CNN predicted class |
| Confidence | CNN softmax probability of that class |
| Severity | derived from the CNN confidence and the affected area |
| Region outline, area, position | OpenCV contours |
| Dirty coverage % | accepted mask pixels ÷ total pixels |
| Cleaning action, tool, passes | rule table keyed on the predicted class |
| Route, distance, mission time | nearest-neighbour simulation |

The dashboard labels the confidence metric **"CNN prediction confidence"**, the
detection table has a **Classifier** column reading `CNN`, and the JSON report
carries `"trained_model_used": true` along with the full probability vector for
every detection. If the model is missing, all three switch back to saying
*heuristic* — the wording always matches what actually ran.

---

## Stain classes

| Class folder | Dashboard label | What it looks like | Cleaning action |
|---|---|---|---|
| `clean` | *(rejected, never shown)* | plain floor, grout lines, grout junctions, glare, shadows, heavy speckle, image-border artifacts | — |
| `mud_dirt` | Mud / Dirt | brown, granular, irregular, with splashes; also faint wide built-up grime | Scrub + Water, 3 passes |
| `liquid_spill` | Liquid Spill | darker, smooth, rounded, with a bright wet rim | Scrub + Suction, 2 passes |
| `grease_oil` | Grease / Oil | dark, glossy, smeared, with specular highlights inside it | Degrease + Scrub, 3 passes |
| `dark_stain` | Dark Stain | deep, matte, set-in discolouration, often with a dried halo | Scrub + Detergent, 3 passes |
| `scattered_dirt` | Scattered Dirt | many small particles across the area; also dry dust or scuff streaks | Vacuum + Light Pass, 2 passes |

`clean` is the class that does the real work. The OpenCV stage cannot help
proposing tile joints, reflections and shadow edges, so the network is trained on
exactly those seven distractor sub-types and learns to veto them. Removing false
positives is a *classification* problem here, not another threshold.

---

## Dataset generation

```bat
python dataset_generator.py --per-class 1000
```

`dataset_generator.py` reuses the same floor and stain primitives that produce
the demo image — `build_tiled_floor`, `blob_alpha`, `apply_stain`,
`apply_lighting` from `sample_generator.py` — so the classifier is trained on the
same visual world the detector is demonstrated on.

Each crop is built by generating an oversized tiled floor, taking a random window
from it, painting a stain (or a distractor), then applying camera conditions.
Variation is introduced in:

| Property | Range |
|---|---|
| Tile colour | 10 base colours (grey, warm grey, cream, beige, blue-grey, near-white), each scaled ±12% |
| Tile size | 0.55× – 1.9× the crop side, so grout spacing varies |
| Grout position | random window; can be forced onto a joint or a junction |
| Floor texture | fine speckle σ 0.4 – 3.4 plus coarse mottling |
| Stain size | radius 18% – 52% of the crop |
| Stain shape | 3 – 14 irregular lobes, softness 0.16 – 0.62, plus directional smearing |
| Stain colour | per-class BGR ranges (brown, neutral dark, warm-tinted film, pale dust) |
| Stain strength | 0.18 – 0.92 alpha, so both fresh and faded soiling appear |
| Lighting | diagonal gradient, a specular highlight, a soft corner shadow |
| Brightness | global gain 0.74 – 1.24, offset ±16 |
| Contrast | 0.78× – 1.26× around the mean |
| Rotation | full 0 – 360°, plus random horizontal flip |
| Scale | 1.0× – 1.18× zoom |
| Blur | Gaussian σ 0 – 1.7 |
| Noise | Gaussian σ 0.5 – 5.5 |
| Compression | JPEG quality 45 – 96 on 55% of crops |
| Position | stain centre jittered within the middle half of the crop |

The result is written as image folders with a **70 / 15 / 15 train / validation /
test split**:

```
dataset/
    train/  clean/ mud_dirt/ liquid_spill/ grease_oil/ dark_stain/ scattered_dirt/
    val/    ... same six classes ...
    test/   ... same six classes ...
    dataset_summary.json
```

The `clean` class is over-sampled by 1.5× because it covers seven visually
different sub-types rather than one. The dataset is **not** committed to this
repository (about 14 MB across 6,500 files); regenerate it with the command
above whenever you need it.

---

## Model performance

### The model

A small VGG-style CNN, defined and trained from scratch in
`train_classifier.py`:

```
input 96x96x3
  Rescaling 1/255
  [Conv3x3 32  → BN → ReLU] x2 → MaxPool
  [Conv3x3 64  → BN → ReLU] x2 → MaxPool
  [Conv3x3 96  → BN → ReLU] x2 → MaxPool
  [Conv3x3 128 → BN → ReLU] x2 → MaxPool
  GlobalAveragePooling → Dropout 0.35
  Dense 128 ReLU       → Dropout 0.35
  Dense 6 softmax
```

| Property | Value |
|---|---|
| Parameters | 481,510 |
| Input size | 96 × 96 RGB |
| Framework | TensorFlow 2.21.0 / Keras 3.15.1 |
| Optimiser | Adam, lr 1e-3, `ReduceLROnPlateau` |
| Loss | sparse categorical cross-entropy |
| Batch size | 64 |
| Best epoch | 27 (early stopping on validation accuracy) |
| Pretrained weights | **none** — trained from scratch, nothing downloaded |
| Model file | `models/stain_classifier.keras`, 5.6 MB |

A global-average-pooling head instead of a large flattened dense layer keeps the
parameter count and the overfitting risk down. Augmentation is applied to the
training split only, on top of the variation already baked into the dataset:
random flips, full 360° rotation, ±10% translation, ±15% zoom, ±20% contrast and
±15% brightness. Floor photographs have no canonical orientation or exposure, so
all of these are label-preserving.

### Results on the held-out synthetic test split

> **96.4% test accuracy on the generated/synthetic test dataset.**
> This figure describes performance on synthetic crops drawn from the same
> generator that produced the training data. It is **not** a measurement of
> real-world accuracy on photographs of real floors, and must not be read as
> one. See [Academic Limitations](#academic-limitations).

975 held-out crops, never seen during training:

| Metric | Value |
|---|---|
| Test accuracy | **96.4%** |
| Precision (macro) | 96.9% |
| Recall (macro) | 96.1% |
| F1-score (macro) | 96.4% |
| F1-score (weighted) | 96.4% |
| Stain-vs-clean F1 | 98.8% |
| Stain-vs-clean precision | 100.0% |
| Stain-vs-clean recall | 97.6% |

The stain-vs-clean row is the operationally important one: it measures the
accept/reject decision on its own, ignoring which stain type was chosen. On this
synthetic test set the model never labelled a `clean` crop as a stain.

### Class-wise performance

| Class | Precision | Recall | F1-score | Test samples |
|---|---|---|---|---|
| `clean` | 0.926 | 1.000 | 0.962 | 225 |
| `mud_dirt` | 0.986 | 0.967 | 0.976 | 150 |
| `liquid_spill` | 1.000 | 0.993 | 0.997 | 150 |
| `grease_oil` | 0.972 | 0.933 | 0.952 | 150 |
| `dark_stain` | 0.936 | 0.973 | 0.954 | 150 |
| `scattered_dirt` | 0.993 | 0.900 | 0.944 | 150 |

### Confusion matrix

Rows are the true class, columns the predicted class:

| | pred `clean` | pred `mud_dirt` | pred `liquid_spill` | pred `grease_oil` | pred `dark_stain` | pred `scattered_dirt` |
|---|---|---|---|---|---|---|
| **true `clean`** | **225** | 0 | 0 | 0 | 0 | 0 |
| **true `mud_dirt`** | 5 | **145** | 0 | 0 | 0 | 0 |
| **true `liquid_spill`** | 0 | 0 | **149** | 0 | 1 | 0 |
| **true `grease_oil`** | 0 | 0 | 0 | **140** | 9 | 1 |
| **true `dark_stain`** | 0 | 0 | 0 | 4 | **146** | 0 |
| **true `scattered_dirt`** | 13 | 2 | 0 | 0 | 0 | **135** |

Reading the errors:

- The largest confusion is **`grease_oil` → `dark_stain`** (9 crops) and back
  (4 crops). Both are dark, and they differ mainly in gloss and edge behaviour,
  which survive poorly under blur and low contrast.
- **`scattered_dirt` → `clean`** (13 crops) is the main recall loss: sparse,
  faint debris on a speckled tile genuinely resembles clean floor.
- No `clean` crop was misread as a stain, which is why the false-positive
  behaviour in the dashboard is good.

These metrics are stored in `models/metrics.json` and rendered live in the
dashboard's **Model performance** tab, alongside the training curves from
`models/training_history.json`.

---

## Retraining

The repository ships a trained model, so retraining is optional. To rebuild it
from scratch:

```bat
python dataset_generator.py --per-class 1500 --seed 42
python train_classifier.py --epochs 60 --batch-size 64
python selftest.py
```

Dataset generation takes about a minute; training takes roughly half an hour on
a typical laptop CPU.

Useful options:

- `--per-class` — total crops per stain class across all splits.
- `--seed` — changes every random draw, giving a genuinely different dataset.
- `--epochs`, `--batch-size`, `--learning-rate`, `--patience` on the trainer.

> **Note.** `train_classifier.py` writes to `models/stain_classifier.keras` via a
> Keras checkpoint callback, overwriting the committed model as soon as
> validation accuracy improves. Copy the existing file aside first if you want
> to keep it.

To add a class: add the folder name to `CLASS_NAMES` in `dataset_generator.py`,
write a painter function for it, register it in `PAINTERS`, add a display name to
`DISPLAY_NAMES` in `train_classifier.py`, and add a matching entry to
`CLEANING_ACTIONS` in `cleaning_engine.py` so the robot knows what to do about
it. Then regenerate and retrain. Nothing in `app.py` needs to change; the
dashboard reads the class list from `models/model_config.json`.

To train on **real** photographs instead of synthetic crops, point
`train_classifier.py --dataset` at image folders using the same six class names
and the same `train/ val/ test/` layout. This is the path from an academic
prototype towards something measurable on real floors.

---

## Screenshots

Screenshots of the running dashboard belong here. To capture them, start the app
with `python -m streamlit run app.py`, press **Load sample floor**, and capture
each tab.

Suggested set, saved into a `docs/screenshots/` folder:

| File | What to capture |
|---|---|
| `01-dashboard.png` | The full dashboard after loading the sample floor: header, the five metric cards and the sidebar. |
| `02-detection.png` | The **Detection** tab: original image beside the AI analysis with boxes, labels and CNN confidences. |
| `03-detection-mask.png` | The anomaly/detection mask view with the filtered-out counts underneath. |
| `04-cleaning-decisions.png` | The **Cleaning decisions** tab: mission summary and the priority-sorted task queue. |
| `05-floor-map.png` | The **Floor map** tab: the 2D zone map with the simulated robot route. |
| `06-reports.png` | The **Reports** tab: download buttons and the JSON preview. |
| `07-model-performance.png` | The **Model performance** tab: accuracy cards, class-wise table and confusion matrix. |
| `08-single-stain.png` | `samples/brown_stain_test.jpg` analysed, showing one detection on the central stain and no border or grout artifacts. |

Then embed them here, for example:

```markdown
![Detection tab](docs/screenshots/02-detection.png)
![Model performance](docs/screenshots/07-model-performance.png)
```

---

## Project folder structure

```
cc1_vision/
│
├── app.py                  Streamlit dashboard (UI only)
├── detector.py             OpenCV candidate stage, filters, drawing
├── ml_classifier.py        CNN runtime: crop, load, batched inference
├── cleaning_engine.py      Stain to cleaning action / priority rules
├── route_simulator.py      Floor map and simulated robot route
├── utils.py                Image IO, validation, reports, formatting
├── sample_generator.py     Synthetic test floor generator
├── dataset_generator.py    Labelled crop dataset builder (6 classes)
├── train_classifier.py     Local CNN training and evaluation
├── selftest.py             Headless end-to-end verification (13 checks)
│
├── requirements.txt        Base dependencies (5 packages)
├── requirements-ml.txt     Adds TensorFlow + scikit-learn
├── run.bat                 One-click Windows launcher
├── README.md               This file
├── .gitignore
│
├── .streamlit/
│   └── config.toml         Interface theme
│
├── models/                 Trained model and its metadata (COMMITTED)
│   ├── stain_classifier.keras    trained CNN, 5.6 MB
│   ├── labels.json               class names in model output order
│   ├── model_config.json         input size, class order, thresholds
│   ├── training_history.json     per-epoch accuracy and loss
│   └── metrics.json              test accuracy, P/R/F1, confusion matrix
│
├── samples/
│   ├── sample_floor.jpg          multi-stain demo floor
│   └── brown_stain_test.jpg      single central stain, regression image
│
├── dataset/                (git-ignored, regenerate with dataset_generator.py)
│   ├── train/ val/ test/         six class folders each
│   └── dataset_summary.json
│
└── reports/                (git-ignored except .gitkeep)
    └── saved PNG / JSON / CSV analysis bundles
```

### What is committed and what is not

| Ignored | Why |
|---|---|
| `.venv/` | Virtual environment, ~2 GB, machine-specific |
| `__pycache__/`, `*.pyc` | Compiled bytecode |
| `.pytest_cache/`, other tool caches | Regenerated automatically |
| `dataset/` | ~14 MB across 6,500 generated files; reproducible in a minute |
| `reports/*` | Analysis output, not source (`.gitkeep` is kept) |
| `cc1_vision.zip` | A redundant archive of this same project |
| `*.log`, `*.tmp`, `logs/` | Temporary files |
| `Thumbs.db`, `.DS_Store`, `Desktop.ini` | OS clutter |

`models/stain_classifier.keras` is **deliberately committed** so the project runs
immediately after cloning, with no training step.

---

## Limitations

**Model and labels**

- The classifier is trained on synthetic data only. The reported 96.4% is a
  synthetic-domain figure, not validated real-world accuracy.
- Six classes cannot cover real soiling. A stain that fits none of them is
  forced into the nearest one, or rejected as clean. A blue drink spill, for
  instance, is most often reported as a dark stain.
- `liquid_spill`, `grease_oil` and `dark_stain` are the classes most easily
  confused with one another; they differ mainly in gloss and edge behaviour,
  which survive poorly under blur and low contrast.
- Confidence is a softmax probability. It is the network's own output, but it is
  not calibrated against real-world frequencies, so a 95% reading does not mean
  95 correct out of 100 on real floors.

**Detection**

- The background model assumes the floor is the dominant surface. Photographs
  containing furniture, feet or large objects will flag those objects as
  anomalies, and the CNN will often accept them as stains.
- Stains wider than roughly a third of the image blend into the background model
  and may be missed or under-segmented.
- Patterned or multi-coloured floors (mosaic, heavy marble veining, printed
  designs) raise the false-positive rate; lower the sensitivity for these.
- Very strong shadows with hard edges can still be reported as dark stains.
- A wet but clean floor and a transparent spill can look identical to this
  method.
- Fragment merging uses a fixed distance rule, so two genuinely separate stains
  closer than 3% of the image can be reported as one.
- The CNN only ever sees regions OpenCV proposed. A stain the anomaly stage
  never flags cannot be recovered by the classifier.
- JPEG compression artifacts can occasionally score just above the acceptance
  threshold; the sidebar slider exists so this can be tuned per image.

**Measurement and simulation**

- Area is measured in pixels. The square-metre and metre figures on the floor
  map depend entirely on the "assumed zone width" slider; there is no real
  calibration.
- Cleaning actions, water levels, pass counts and mission times are a simulation
  of robot behaviour, not measurements from hardware, and the hazard notes are
  not a safety assessment.

---

## AI Assistance Disclosure

**This academic prototype was created with significant assistance from
generative AI tools, including Anthropic's Claude and OpenAI's ChatGPT.** That
assistance was substantial and is not incidental to the result, so it is stated
here plainly rather than hidden or minimised.

AI tools assisted with:

- **Code generation** — drafting the module structure and much of the
  implementation across `app.py`, `detector.py`, `ml_classifier.py`,
  `dataset_generator.py`, `train_classifier.py`, `cleaning_engine.py`,
  `route_simulator.py`, `utils.py`, `sample_generator.py` and `selftest.py`.
- **Debugging** — diagnosing and fixing pipeline defects, including a fragment
  ordering bug that discarded small stain pieces before they could be merged, a
  shadow filter that did not check darkness, and an environment problem where
  TensorFlow had no wheels for the installed Python version.
- **TensorFlow/Keras CNN implementation** — the CNN architecture, the
  augmentation pipeline, the training loop, the callbacks, the evaluation code
  and the metrics/serialisation format.
- **OpenCV processing** — the local background model, the anomaly fusion, grout
  and glare suppression, morphology, contour handling, fragment merging and the
  geometric false-positive filters.
- **Streamlit UI development** — dashboard layout, the metric cards, the tab
  structure, the sidebar controls and the model-performance views.
- **Testing** — the design and implementation of the 13-check headless
  self-test, including the single-stain regression test and the bad-input
  handling checks.
- **Documentation** — this README, the module and function docstrings, and the
  inline explanatory comments.

**The final project was run, tested, reviewed and demonstrated by the student.**
The student set the requirements and scope, directed the work and the design
decisions, reviewed the generated code, executed the dataset generation and the
model training locally, ran the self-test suite, verified the dashboard
behaviour in the browser, validated the detection results against known test
images, and is responsible for the submitted work and for the claims made about
it in this document.

---

## Academic Limitations

This section states the boundaries of what this project demonstrates, so its
results are not overstated.

**Much of the model training data is synthetically generated.** The entire
training, validation and test dataset is drawn programmatically by
`dataset_generator.py` using NumPy and OpenCV. No real photographs of real dirty
floors were used to train or evaluate the classifier. The generator defines the
problem: the network can only learn distinctions the painter functions actually
draw, and real grease, real mud and a real set-in stain differ in ways this
generator does not model.

**The reported accuracy is a synthetic-domain figure.** The test split is
genuinely held out and never trained on, so **96.4% test accuracy on the
generated/synthetic test dataset** is a valid measurement — of performance on
synthetic data from the same generator. It says nothing reliable about accuracy
on real floors. Synthetic classes are also cleanly separated by construction,
which makes the test set easier than reality.

**Performance on real-world floors may differ, and will most likely be worse.**
Real environments introduce conditions this project simulates only crudely or
not at all: wood, carpet, patterned vinyl, mosaic, polished concrete and marble
veining; motion blur and rolling shutter; mixed colour temperature; deep shadow
and heavy over-exposure; reflections of people, furniture and windows; and
soiling that fits none of the six classes. A meaningful real-world claim would
require a dataset of annotated real floor photographs and a fresh evaluation
on it.

**This is an academic prototype.** It was built to demonstrate a vision and
decision architecture — OpenCV localisation combined with CNN classification,
feeding a cleaning-decision and route-simulation layer — and to show that the
pipeline is correctly wired end to end. It is a coursework artefact, not
engineered or validated software.

**It is not a production autonomous cleaning system.** No robot is controlled
and no hardware is involved. The cleaning actions, tools, water levels, pass
counts, priorities, routes and mission times are a *simulation* of how such a
robot might behave. They are not chemical, hygiene or slip-safety assessments,
they have not been validated against any cleaning standard, and they should not
be used to make real cleaning, safety or purchasing decisions.

---

## Troubleshooting

**The sidebar says `Classifier: Heuristic rules (fallback)`**
The trained model is not being used. The **Model status** row gives the reason:

| Message | Fix |
|---|---|
| `No trained model found in models/.` | Confirm `models/stain_classifier.keras` was cloned; otherwise `python dataset_generator.py` then `python train_classifier.py` |
| `TensorFlow is not installed (see requirements-ml.txt).` | `pip install -r requirements-ml.txt`, on Python 3.10 – 3.12 |
| `Model could not be loaded: ...` | The file is corrupt or was written by an incompatible Keras version. Retrain it. |

**`No matching distribution found for tensorflow`**
Your Python is too new. TensorFlow ships wheels for 3.10 – 3.12; see *A note on
the Python version* under [Installation](#installation).

**`'streamlit' is not recognized as an internal or external command`**
The Scripts folder is not on your PATH, or the virtual environment is not
active. Start the app this way instead:

```bat
.venv\Scripts\activate
python -m streamlit run app.py
```

**`ModuleNotFoundError: No module named 'detector'`** or **`FileNotFoundError: app.py`**
You are in the wrong working directory. All modules import each other by name,
so run from the folder containing `app.py`.

**`Dataset split(s) [...] not found`**
`train_classifier.py` was run before the dataset existed. Run
`python dataset_generator.py` first.

**The sample image is missing**
Regenerate it: `python sample_generator.py`. The **Load sample floor** button
also generates the floor in memory, so the app works with an empty `samples/`
folder.

**Port 8501 is already in use**
`python -m streamlit run app.py --server.port 8502`

**Nothing is detected in my photo**
Raise **Detection Sensitivity**, lower **Minimum Detection Area**, and lower
**Minimum CNN stain probability** under Advanced filtering.

**Everything is detected in my photo**
Lower the sensitivity, raise the minimum area, raise **Minimum CNN stain
probability**, and confirm the three suppression filters are enabled. Patterned
floors need a sensitivity around 20–35.

**Analysis feels slow**
Lower the Analysis Resolution to 960 or 640 in the sidebar. Typical runtime is
around half a second at 1280 px on an ordinary laptop CPU, plus a few seconds
once at startup while Keras loads the model.

---

## Future scope

- Retrain the CNN on real annotated floor photographs and re-measure on them.
- Calibrate the confidence output (temperature scaling) against a real test set.
- Add a segmentation model (U-Net or similar) for pixel-accurate stain masks.
- Camera calibration and homography so pixel areas become true square metres.
- Multi-frame fusion, so a moving robot confirms a stain across several views.
- Before/after verification: re-scan a cleaned area and confirm removal.
- Battery, water-tank and coverage-path planning instead of nearest neighbour.
- Logging detections over time to build a heat map of chronically dirty zones.
- ROS 2 node wrapping the detector for real robot integration.

---

## Author's note

The project started as a demonstration that a useful, honest dirt-detection
layer can be built from classical computer vision alone. It now pairs that layer
with a convolutional network trained from scratch on this machine, which is what
removed the grout-line, corner and texture false positives that no amount of
threshold tuning had fixed: deciding "is this actually a stain?" turned out to
be a classification problem, not another rule.

The commitment to honesty did not change with the addition of machine learning.
The training data is synthetic, the reported accuracy is a synthetic-domain
figure, and this README, the dashboard and the JSON report all say so. A model
trained on data it generated itself is a working demonstration of an
architecture, not evidence of real-world performance, and this project does not
claim otherwise.

---

*Academic concept inspired by autonomous cleaning systems. Not affiliated with,
endorsed by, or connected to any robot manufacturer. No physical robot is
controlled.*
