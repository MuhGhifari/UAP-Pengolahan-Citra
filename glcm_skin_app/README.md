# GLCM Skin Disease Classifier (glcm_skin_app)

This project implements a complete pipeline to extract Gray-Level Co-occurrence Matrix (GLCM) texture features from dermatoscopic / clinical skin images, train a Random Forest classifier, and serve a lightweight Flask web demo for inference and history tracking.

The goal is a self-contained, reproducible final-project application you can run locally to demonstrate texture-based classification using the SkinDisease dataset.

---

**Contents:**

- **Project:** small Flask app + training pipeline
- **Feature method:** GLCM (4 directions, configurable gray levels)
- **Classifier:** scikit-learn RandomForestClassifier
- **Persistence:** Pickled model artifact, JSON metrics, SQLite history
- **UI:** simple upload page and results/history pages served by Flask

---

**Quick overview**

1. Extract features from images with `features.py` using GLCM (texture descriptors per direction).
2. Train the classifier with `train.py`. This writes `model/glcm_skin_model.pkl` and `model/metrics.json` and saves a confusion image to `static/`.
3. Start the demo with `app.py` and open the web UI to upload images and see predictions.

---

**Repository layout (important files)**

- `glcm_skin_app/app.py` — Flask application and endpoints for upload, predict, and history.
- `glcm_skin_app/config.py` — central paths and configuration values.
- `glcm_skin_app/features.py` — GLCM feature extraction and helper utilities.
- `glcm_skin_app/train.py` — dataset discovery, training loop, metrics, and artifact export.
- `glcm_skin_app/requirements.txt` — Python dependencies to install in a venv.
- `glcm_skin_app/model/` — artifact output (pickle model, metrics.json, sqlite DB created by the app).
- `glcm_skin_app/static/` — static assets including `confusion_matrix.png` produced by training and web CSS.
- `glcm_skin_app/templates/` — Jinja2 templates for UI pages.

---

**Requirements**

- Python 3.8+ (tested with Python 3.10–3.14 in various environments). Use your system `python3` or a virtual environment.
- Key Python packages (see `requirements.txt`): `numpy`, `Pillow`, `scikit-learn`, `Flask`. Optionally `matplotlib` may appear but training uses a Pillow renderer to avoid environment recursion issues.

---

**Installation (copy & paste)**

Open a terminal and run:

```bash
cd "/Users/muhghifari/Documents/TA Pengolahan Citra/glcm_skin_app"
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# On Windows (PowerShell): .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- If you already use a global environment, prefer a project venv to keep dependencies isolated.
- If any dependency fails to install, try `pip install --no-cache-dir <package>` or share the error for assistance.

---

**Dataset layout expected by `train.py`**

`train.py` expects a folder structure like the SkinDisease dataset:

```
data/SkinDisease/SkinDisease/train/<class_name>/*.jpg
data/SkinDisease/SkinDisease/test/<class_name>/*.jpg
```

You can pass explicit folders to `train.py --train-dir` and `--test-dir` if your layout differs.

---

**Training quick-start & options**

Quick smoke test (fast; limit images per class):

```bash
python train.py --max-images-per-class 5
```

Full training:

```bash
python train.py
```

Important flags (see `train.py --help`):

- `--train-dir` / `--test-dir`: override dataset paths
- `--image-size`: resize images prior to quantization (default used in code)
- `--levels`: number of gray levels to quantize images to (affects GLCM size)
- `--trees`: number of trees for the Random Forest
- `--max-images-per-class`: quick debugging limit

Outputs written by training (to `glcm_skin_app/model` by default):

- `glcm_skin_model.pkl` — pickled dict containing the trained model and metadata (model, class names, config, feature names, timestamp).
- `metrics.json` — accuracy, precision, recall, f1, dataset sizes, class list.
- `predictions.sqlite3` — (created by the Flask app on first run) stores prediction history; training may also use an ad-hoc DB for evaluation depending on scripts.
- `static/confusion_matrix.png` — confusion matrix rendered as PNG (Pillow renderer to avoid matplotlib recursion problems).

---

**How the feature extraction works (GLCM summary)**

GLCM (Gray-Level Co-occurrence Matrix) is a second-order statistical method that captures how often pairs of pixel intensity values (i, j) occur at a given relative position (offset) in an image. Texture descriptors computed from the GLCM often include contrast, dissimilarity, homogeneity, energy, correlation and angular second moment — these describe different aspects of texture.

Implementation notes in `features.py`:

- Input images are opened with Pillow and converted to grayscale.
- Images are resized to a consistent `image_size` for reproducible statistics.
- Grayscale quantization reduces intensity values to `levels` bins (e.g., 8 or 16) — this limits the size of the GLCM and reduces sensitivity to noise.
- GLCMs are computed for 4 directions (0°, 45°, 90°, 135°) with distance=1 by default, and a set of texture metrics is computed per direction.
- The final feature vector is the concatenation of metrics across directions (e.g., 7 metrics × 4 directions = 28 features). The exact metrics and ordering are exported in the artifact metadata as `feature_names`.

Why this approach:

- GLCM texture features are simple, deterministic, and interpretable — a good fit for classical image processing courses and demonstration projects.
- Using a Random Forest on top of these features provides robustness and easy inspection of feature importance.

---

**Modeling & training details (`train.py`)**

- The script discovers classes by subfolders under the train/test directories and loads images (optionally limiting per-class counts for quick experimentation).
- For each image, it extracts the GLCM feature vector using `features.extract_glcm_features()`.
- Features are stacked into an `X` array and labels into `y`.
- A `RandomForestClassifier` (scikit-learn) with `class_weight='balanced_subsample'` is trained; hyperparameters can be changed via CLI.
- The script computes common metrics (accuracy, precision, recall, f1) and a confusion matrix. Instead of using `matplotlib` to draw the confusion matrix (which caused recursion issues in some environments), the project creates a PNG using Pillow drawing primitives for maximum compatibility.
- The resulting artifacts are saved to `glcm_skin_app/model/`.

---

**Web app (`app.py`) behavior**

- On startup, `app.py` ensures directories exist (uploads, model, static) via `config.py`.
- If the pickled model artifact exists, the app loads it and uses it to classify incoming uploads. If it does not exist, the index page will prompt you to run `train.py` first.
- `POST /analyze` accepts an uploaded image file, extracts the GLCM features with the same quantization and size parameters used during training (these are stored in the artifact metadata and applied at inference), performs prediction, and writes a history record into an SQLite database in `glcm_skin_app/model/predictions.sqlite3`.
- Results page shows: predicted label, confidence, and top-K predictions. The `history` route lists prior uploads and results.

Security notes (demo-level):

- This app is intended for local demonstration and is not production hardened. It accepts image uploads — do not expose it to untrusted networks.
- If you deploy, add authentication, input validation, rate limiting, and run behind a secure server.

---

**Usage examples**

Start the app after training:

```bash
python app.py
# then open http://127.0.0.1:8000
```

Upload a sample image on the web UI or use `curl` to test (replace `path/to/image.jpg`):

```bash
curl -F "file=@path/to/image.jpg" http://127.0.0.1:8000/analyze
```

---

**Troubleshooting**

- If `app.py` prints "Model artifact not found" or the web UI tells you to train first: run `python train.py` and verify `model/glcm_skin_model.pkl` exists.
- If training crashes while plotting the confusion matrix: ensure dependencies were installed; this repo uses a pure-Pillow renderer to avoid known matplotlib recursion failures.
- If predictions are slightly different between train/infer runs, verify the `levels` (quantization) and `image-size` are the same for both phases — these values are saved to the artifact but mismatches can occur if you modified code manually.
- If you encounter memory or performance issues: reduce `--max-images-per-class` for quick experiments, or increase system RAM / use a machine with more cores for larger training.

---

**Evaluation tips & next steps**

- Use `--max-images-per-class` during development to iterate quickly.
- After getting satisfactory results, run full training on the entire dataset without the limit.
- Consider cross-validation, class-wise sample balancing, or more advanced texture descriptors (LBP, wavelets) if GLCM features alone are insufficient.

---

**Files to inspect for implementation details**

- `features.py` — GLCM implementation and feature names.
- `train.py` — data ingestion, training, metrics, and artifact creation.
- `app.py` — upload handling, prediction, and history persistence.

If you want, I can also:

- remove unused `matplotlib` from `requirements.txt` to avoid confusion, or keep it if you plan to plot locally;
- run a quick smoke training and start the Flask app here and report back the results.

---

**License & attribution**

This project is for educational use in your final project. If you reuse external code or papers, cite them appropriately in your report.

---

If you want more details added to any section (math derivation of GLCM metrics, code walkthrough with annotated snippets, or a printable PDF summary for the final report), tell me which and I'll extend the README accordingly.
# GLCM Skin Analyzer

This is a final-project app for image processing coursework. It uses Gray-Level Co-occurrence Matrix texture features to classify the skin-disease dataset under `data/SkinDisease/SkinDisease`.

## What it does

- Converts each image to grayscale.
- Extracts GLCM features from four directions.
- Trains a Random Forest classifier.
- Serves a Flask upload app that predicts the most likely skin-disease class.

## Dataset expected

Keep the current directory layout:

```text
data/SkinDisease/SkinDisease/
├── train/
│   ├── Acne/
│   ├── Actinic_Keratosis/
│   └── ...
└── test/
    ├── Acne/
    ├── Actinic_Keratosis/
    └── ...
```

The class folder names become the labels.

## Install

Use Python 3.11 or newer.

```bash
cd "/Users/muhghifari/Documents/TA Pengolahan Citra/glcm_skin_app"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Run the full dataset:

```bash
python train.py
```

For a quick smoke test, limit the number of images per class:

```bash
python train.py --max-images-per-class 5
```

The script writes:

- `model/glcm_skin_model.pkl`
- `model/metrics.json`
- `static/confusion_matrix.png`

## Run the app

```bash
python app.py
```

Open `http://127.0.0.1:8000`.

## Note for your presentation

GLCM is a texture descriptor, so it is a good fit for explaining local structure, but it is not a medical diagnosis tool. Use the app to show how image features, classifiers, and evaluation work together.
