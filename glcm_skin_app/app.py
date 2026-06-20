from __future__ import annotations

import json
import os
import pickle
import sqlite3
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_EXTENSIONS,
    ARTIFACT_PATH,
    DATABASE_PATH,
    MAX_UPLOAD_SIZE,
    METRICS_PATH,
    UPLOAD_DIR,
    ensure_directories,
)
from features import GLCMConfig, InvalidImageError, extract_glcm_features


DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def create_app(test_config: dict | None = None) -> Flask:
    ensure_directories()
    application = Flask(__name__)
    application.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "development-secret-change-me"),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_SIZE,
        UPLOAD_FOLDER=str(UPLOAD_DIR),
    )
    if test_config:
        application.config.update(test_config)
    initialize_database()
    register_routes(application)
    register_error_handlers(application)
    return application


def database_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with database_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                top_predictions TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC)"
        )


def read_metrics() -> dict | None:
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@lru_cache(maxsize=1)
def load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError("The model is not trained yet. Run python train.py first.")
    with ARTIFACT_PATH.open("rb") as handle:
        return pickle.load(handle)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def humanize_label(label: str) -> str:
    return label.replace("_", " ").replace("/", " ").strip().title()


def classify_image(source: bytes | str | Path) -> dict:
    artifact = load_artifact()
    model = artifact["model"]
    config = artifact.get("config") or {}
    glcm_config = GLCMConfig(
        image_size=int(config.get("image_size", 128)),
        levels=int(config.get("levels", 16)),
    )
    features = extract_glcm_features(source, config=glcm_config).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    class_names = list(model.classes_)
    best_index = int(np.argmax(probabilities))
    ranking = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        {
            "class_name": class_names[index],
            "label": humanize_label(class_names[index]),
            "probability": float(probabilities[index]),
        }
        for index in ranking
    ]
    return {
        "class_name": class_names[best_index],
        "label": humanize_label(class_names[best_index]),
        "confidence": float(probabilities[best_index] * 100),
        "top_predictions": top_predictions,
        "class_count": len(class_names),
        "feature_count": len(artifact.get("feature_names", [])),
    }


def register_routes(application: Flask) -> None:
    @application.get("/")
    def index():
        with database_connection() as connection:
            recent = connection.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        return render_template(
            "index.html",
            model_ready=ARTIFACT_PATH.exists(),
            metrics=read_metrics(),
            recent=recent,
            total_predictions=total,
        )

    @application.post("/analyze")
    def analyze():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            flash("Choose an image before running the analysis.", "danger")
            return redirect(url_for("index"))
        if not allowed_file(uploaded.filename):
            flash("Supported formats are JPG, PNG, WebP, and BMP.", "danger")
            return redirect(url_for("index"))

        original_name = secure_filename(uploaded.filename)
        extension = Path(original_name).suffix.lower()
        identifier = uuid.uuid4().hex
        stored_name = f"{identifier}{extension}"
        destination = UPLOAD_DIR / stored_name

        try:
            image_bytes = uploaded.read()
            result = classify_image(image_bytes)
            destination.write_bytes(image_bytes)
        except InvalidImageError as error:
            flash(str(error), "danger")
            return redirect(url_for("index"))
        except FileNotFoundError as error:
            flash(str(error), "warning")
            return redirect(url_for("index"))
        except Exception:
            destination.unlink(missing_ok=True)
            application.logger.exception("Skin analysis failed")
            flash("The image could not be analyzed. Please try another file.", "danger")
            return redirect(url_for("index"))

        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with database_connection() as connection:
            connection.execute(
                """
                INSERT INTO predictions (filename, predicted_class, confidence, top_predictions, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    original_name,
                    result["label"],
                    result["confidence"],
                    json.dumps(result["top_predictions"]),
                    created_at,
                ),
            )

        return render_template(
            "result.html",
            image_filename=stored_name,
            original_filename=original_name,
            predicted_label=result["label"],
            predicted_class=result["class_name"],
            confidence=result["confidence"],
            top_predictions=result["top_predictions"],
            class_count=result["class_count"],
            feature_count=result["feature_count"],
        )

    @application.get("/history")
    def history():
        page = max(request.args.get("page", 1, type=int), 1)
        per_page = 12
        with database_connection() as connection:
            total = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (per_page, (page - 1) * per_page),
            ).fetchall()
        pages = max(1, (total + per_page - 1) // per_page)
        if total and page > pages:
            return redirect(url_for("history", page=pages))
        return render_template("history.html", predictions=rows, page=page, pages=pages, total=total)

    @application.get("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        return send_from_directory(application.config["UPLOAD_FOLDER"], filename)

    @application.get("/about")
    def about():
        return render_template(
            "error.html",
            code=200,
            message="GLCM measures how often gray levels appear together across local neighborhoods. This project uses those texture descriptors to classify skin disease images.",
        )


def register_error_handlers(application: Flask) -> None:
    @application.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        flash("The image exceeds the 10 MB upload limit.", "danger")
        return redirect(url_for("index"))

    @application.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", code=404, message="That page does not exist."), 404

    @application.errorhandler(500)
    def server_error(_error):
        return render_template("error.html", code=500, message="An unexpected server error occurred."), 500


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
