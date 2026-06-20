from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
DATASET_ROOT = WORKSPACE_DIR / "data" / "SkinDisease" / "SkinDisease"

MODEL_DIR = PROJECT_DIR / "model"
UPLOAD_DIR = PROJECT_DIR / "uploads"
STATIC_DIR = PROJECT_DIR / "static"
TEMPLATE_DIR = PROJECT_DIR / "templates"

ARTIFACT_PATH = MODEL_DIR / "glcm_skin_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = STATIC_DIR / "confusion_matrix.png"
DATABASE_PATH = MODEL_DIR / "predictions.sqlite3"

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}


def ensure_directories() -> None:
    for directory in (MODEL_DIR, UPLOAD_DIR, STATIC_DIR, TEMPLATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
