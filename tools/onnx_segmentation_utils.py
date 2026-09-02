"""Shared BlueFace ONNX input utilities for INT8 calibration and evaluation."""
from pathlib import Path

import cv2
import numpy as np

BLUEFACE_MEAN = np.array((0.46962251, 0.4464104, 0.40718787), dtype=np.float32)
BLUEFACE_STD = np.array((0.27469736, 0.27012361, 0.28515933), dtype=np.float32)


def parse_hw(value):
    """Parse an H,W string used by the fixed-size ONNX export."""
    try:
        height, width = (int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--input-size must be H,W, for example 512,512") from error
    if height <= 0 or width <= 0:
        raise ValueError("--input-size values must be positive")
    return height, width


def resolve_path(dataset_root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(dataset_root) / path


def read_annotation_pairs(dataset_root, annotation_path):
    """Read BlueFace ``image_path,label_path`` annotation rows."""
    annotation_path = resolve_path(dataset_root, annotation_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    pairs = []
    for line_number, line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            image_path, label_path = (item.strip() for item in line.split(",", 1))
        except ValueError as error:
            raise ValueError(f"Invalid row {line_number} in {annotation_path}: {line!r}") from error
        image_path = resolve_path(dataset_root, image_path)
        label_path = resolve_path(dataset_root, label_path)
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing image or label in row {line_number}: {image_path}, {label_path}")
        pairs.append((image_path, label_path))

    if not pairs:
        raise ValueError(f"No image/label pairs found in {annotation_path}")
    return pairs


def preprocess_image(image_path, input_size):
    """Load an RGB image and reproduce ``BlueFaceDataset.ToTensor`` preprocessing."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = input_size
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32).transpose(2, 0, 1) / 255.0
    image = (image - BLUEFACE_MEAN[:, None, None]) / BLUEFACE_STD[:, None, None]
    return np.ascontiguousarray(image[None, ...], dtype=np.float32)


def load_resized_label(label_path, input_size):
    """Load a label image and align it to the fixed ONNX input size."""
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise ValueError(f"Unable to read label: {label_path}")
    height, width = input_size
    label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
    return label.astype(np.int64, copy=False)
