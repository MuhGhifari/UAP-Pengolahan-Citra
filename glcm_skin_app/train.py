from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import ARTIFACT_PATH, CONFUSION_MATRIX_PATH, DATASET_ROOT, METRICS_PATH, ensure_directories
from features import GLCMConfig, describe_feature_vector, extract_glcm_features


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    split: str


def discover_samples(split_dir: Path, split_name: str) -> list[Sample]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {split_dir}")

    samples: list[Sample] = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(Sample(path=path, label=class_dir.name, split=split_name))

    if not samples:
        raise ValueError(f"No images found in {split_dir}.")
    return samples


def limit_samples(samples: list[Sample], max_images_per_class: int | None, seed: int) -> list[Sample]:
    if max_images_per_class is None:
        return samples

    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = random.Random(seed)
    limited: list[Sample] = []
    for label in sorted(grouped):
        bucket = grouped[label]
        if len(bucket) > max_images_per_class:
            bucket = rng.sample(bucket, max_images_per_class)
        limited.extend(sorted(bucket, key=lambda item: item.path.name))
    return limited


def build_matrix(samples: list[Sample], image_size: int, levels: int) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    config = GLCMConfig(image_size=image_size, levels=levels)

    for sample in samples:
        try:
            features.append(extract_glcm_features(sample.path, config=config))
            labels.append(sample.label)
        except ValueError as exc:
            print(f"Skipping {sample.path}: {exc}")

    if not features:
        raise ValueError("No valid images could be processed.")

    return np.vstack(features), np.asarray(labels)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str]) -> None:
    cell_size = 32 if len(class_names) <= 16 else max(18, 720 // len(class_names))
    label_margin = 260
    canvas_size = label_margin + (cell_size * len(class_names)) + 40
    image = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    max_value = int(matrix.max()) if matrix.size else 0
    for row_index, actual_name in enumerate(class_names):
        y = label_margin + row_index * cell_size
        draw.text((16, y + cell_size / 2 - 4), actual_name, fill=(35, 35, 35), font=font)
        for column_index, predicted_name in enumerate(class_names):
            x = label_margin + column_index * cell_size
            value = int(matrix[row_index, column_index])
            intensity = 240 if max_value == 0 else int(245 - 175 * (value / max_value))
            fill = (intensity, 230, intensity)
            draw.rectangle([x, y, x + cell_size, y + cell_size], fill=fill, outline=(185, 205, 190))

            text = str(value)
            text_box = draw.textbbox((0, 0), text, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            draw.text(
                (x + (cell_size - text_width) / 2, y + (cell_size - text_height) / 2 - 1),
                text,
                fill=(30, 30, 30),
                font=font,
            )

    for column_index, class_name in enumerate(class_names):
        x = label_margin + column_index * cell_size + cell_size / 2
        draw.text((x - 8, 16), class_name, fill=(35, 35, 35), font=font)

    draw.text((label_margin + 10, canvas_size - 24), "Predicted", fill=(35, 35, 35), font=font)
    draw.text((16, 18), "Actual", fill=(35, 35, 35), font=font)
    image.save(CONFUSION_MATRIX_PATH)


def train(args: argparse.Namespace) -> dict:
    ensure_directories()
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_samples = limit_samples(
        discover_samples(args.train_dir, "train"), args.max_images_per_class, args.seed
    )
    test_samples = limit_samples(
        discover_samples(args.test_dir, "test"), args.max_images_per_class, args.seed
    )

    class_names = sorted({sample.label for sample in train_samples} | {sample.label for sample in test_samples})
    if len(class_names) < 2:
        raise ValueError("At least two classes are required for training.")

    x_train, y_train = build_matrix(train_samples, args.image_size, args.levels)
    x_test, y_test = build_matrix(test_samples, args.image_size, args.levels)

    model = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=args.seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, predictions, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, predictions, average="weighted", zero_division=0))
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=class_names)

    per_class_accuracy: dict[str, float] = {}
    for class_name in class_names:
        class_mask = y_test == class_name
        if np.any(class_mask):
            per_class_accuracy[class_name] = float(accuracy_score(y_test[class_mask], predictions[class_mask]))

    top_predictions = []
    for row in probabilities[: min(5, len(probabilities))]:
        ranking = np.argsort(row)[::-1][:3]
        top_predictions.append(
            [
                {"class_name": model.classes_[index], "probability": float(row[index])}
                for index in ranking
            ]
        )

    config = {
        "image_size": args.image_size,
        "levels": args.levels,
    }
    artifact = {
        "model": model,
        "class_names": class_names,
        "config": config,
        "feature_names": describe_feature_vector(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with ARTIFACT_PATH.open("wb") as handle:
        pickle.dump(artifact, handle)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "per_class_accuracy": per_class_accuracy,
        "class_names": class_names,
        "train_size": len(train_samples),
        "test_size": len(test_samples),
        "image_size": args.image_size,
        "levels": args.levels,
        "trees": args.trees,
        "report": report,
        "top_predictions_preview": top_predictions,
        "trained_at": artifact["trained_at"],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion_matrix(matrix, class_names)

    print(json.dumps({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "train_size": len(train_samples),
        "test_size": len(test_samples),
        "classes": len(class_names),
    }, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GLCM-based skin disease classifier.")
    parser.add_argument("--train-dir", type=Path, default=DATASET_ROOT / "train")
    parser.add_argument("--test-dir", type=Path, default=DATASET_ROOT / "test")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--levels", type=int, default=16)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Optional quick-test limit per class for faster experimentation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Training failed: {error}") from error
