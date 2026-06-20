from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


DEFAULT_IMAGE_SIZE = 128
DEFAULT_LEVELS = 16
ANGLES = (("0deg", (0, 1)), ("45deg", (-1, 1)), ("90deg", (-1, 0)), ("135deg", (-1, -1)))
FEATURE_NAMES = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "asm",
    "energy",
    "correlation",
    "entropy",
)


class InvalidImageError(ValueError):
    pass


@dataclass(frozen=True)
class GLCMConfig:
    image_size: int = DEFAULT_IMAGE_SIZE
    levels: int = DEFAULT_LEVELS


def _open_image(source: bytes | str | Path) -> Image.Image:
    try:
        if isinstance(source, bytes):
            stream = BytesIO(source)
            with Image.open(stream) as image:
                return image.convert("L").copy()
        with Image.open(source) as image:
            return image.convert("L").copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The selected file is not a valid image.") from exc


def _resize(image: Image.Image, image_size: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image)
    return image.resize((image_size, image_size), resampling.LANCZOS)


def _quantize(image: Image.Image, levels: int) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    quantized = (array.astype(np.uint16) * levels) // 256
    return np.clip(quantized, 0, levels - 1).astype(np.uint8)


def _shift_views(array: np.ndarray, dr: int, dc: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = array.shape

    if dr >= 0:
        source_rows = slice(0, rows - dr)
        target_rows = slice(dr, rows)
    else:
        source_rows = slice(-dr, rows)
        target_rows = slice(0, rows + dr)

    if dc >= 0:
        source_cols = slice(0, cols - dc)
        target_cols = slice(dc, cols)
    else:
        source_cols = slice(-dc, cols)
        target_cols = slice(0, cols + dc)

    return array[source_rows, source_cols], array[target_rows, target_cols]


def _glcm(array: np.ndarray, dr: int, dc: int, levels: int) -> np.ndarray:
    source, target = _shift_views(array, dr, dc)
    matrix = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(matrix, (source.ravel(), target.ravel()), 1)
    matrix += matrix.T
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def _texture_features(matrix: np.ndarray) -> np.ndarray:
    indices = np.arange(matrix.shape[0], dtype=np.float64)
    i, j = np.meshgrid(indices, indices, indexing="ij")
    diff = i - j

    contrast = float(np.sum((diff**2) * matrix))
    dissimilarity = float(np.sum(np.abs(diff) * matrix))
    homogeneity = float(np.sum(matrix / (1.0 + diff**2)))
    asm = float(np.sum(matrix**2))
    energy = float(np.sqrt(asm))

    mu_i = float(np.sum(i * matrix))
    mu_j = float(np.sum(j * matrix))
    sigma_i = float(np.sqrt(np.sum(((i - mu_i) ** 2) * matrix)))
    sigma_j = float(np.sqrt(np.sum(((j - mu_j) ** 2) * matrix)))
    if sigma_i > 0 and sigma_j > 0:
        correlation = float(np.sum(((i - mu_i) * (j - mu_j) * matrix)) / (sigma_i * sigma_j))
    else:
        correlation = 0.0

    non_zero = matrix > 0
    entropy = float(-np.sum(matrix[non_zero] * np.log2(matrix[non_zero]))) if np.any(non_zero) else 0.0
    return np.asarray([contrast, dissimilarity, homogeneity, asm, energy, correlation, entropy], dtype=np.float32)


def extract_glcm_features(source: bytes | str | Path, config: GLCMConfig | None = None) -> np.ndarray:
    config = config or GLCMConfig()
    image = _resize(_open_image(source), config.image_size)
    quantized = _quantize(image, config.levels)

    vectors = []
    for _angle_name, (dr, dc) in ANGLES:
        matrix = _glcm(quantized, dr, dc, config.levels)
        vectors.append(_texture_features(matrix))
    return np.concatenate(vectors)


def describe_feature_vector() -> list[str]:
    names: list[str] = []
    for angle_name, _offset in ANGLES:
        for feature_name in FEATURE_NAMES:
            names.append(f"{feature_name}_{angle_name}")
    return names
