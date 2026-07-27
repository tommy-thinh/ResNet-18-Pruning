#!/usr/bin/env python
"""Print dataset statistics for the multi-label MNIST splits."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def load_data_pt(path: str | os.PathLike[str]):
    """
    Load one saved .pt split and convert it to NumPy arrays.

    This matches the data loader used in `resnet-18.py`.
    """
    data = torch.load(path, map_location="cpu")

    # Convert [N, H, W] uint8 to [N, H, W, 1] float32 in [0, 1].
    images = data["images"].numpy().astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1)

    # ResNet expects 3-channel inputs, so replicate the grayscale channel.
    images = np.repeat(images, repeats=3, axis=-1)

    # Multi-label targets.
    labels = data["labels"].numpy().astype("float32")

    # Extra metadata.
    meta = {
        "center_labels": data["center_labels"].numpy().astype("int64"),
        "count_labels": data["count_labels"].numpy().astype("int64"),
        "num_digits": data["num_digits"].numpy().astype("int64"),
        "all_digit_labels": data["all_digit_labels"].numpy().astype("int64"),
        "all_positions": data["all_positions"].numpy().astype("int64"),
        "label_type": data.get("label_type", None),
    }

    return images, labels, meta


def inspect_dataset(name, images, labels, meta):
    print("=" * 60)
    print(f"{name} SET")
    print("=" * 60)

    # --------------------------------------------------
    # Basic information
    # --------------------------------------------------
    print(f"Images shape : {images.shape}")
    print(f"Labels shape : {labels.shape}")
    print(f"Total images : {len(images)}")

    # --------------------------------------------------
    # Image statistics
    # --------------------------------------------------
    print("\nImage statistics")
    print(f"Mean : {images.mean():.4f}")
    print(f"Std  : {images.std():.4f}")
    print(f"Min  : {images.min():.4f}")
    print(f"Max  : {images.max():.4f}")

    # --------------------------------------------------
    # Multi-hot label statistics
    # --------------------------------------------------
    print("\nDigit frequencies")

    class_counts = labels.sum(axis=0).astype(int)
    class_percent = class_counts / len(labels) * 100

    for digit in range(10):
        print(
            f"Digit {digit}: "
            f"{class_counts[digit]:6d} "
            f"({class_percent[digit]:6.2f}%)"
        )

    # --------------------------------------------------
    # Number of positive labels
    # --------------------------------------------------
    positives = labels.sum(axis=1)

    print("\nPositive labels per image")
    print(f"Mean : {positives.mean():.2f}")
    print(f"Min  : {positives.min():.0f}")
    print(f"Max  : {positives.max():.0f}")

    values, counts = np.unique(positives, return_counts=True)
    print("\nDistribution:")
    for v, c in zip(values, counts):
        print(f"{int(v)} digits : {c}")

    # --------------------------------------------------
    # Actual number of digits
    # --------------------------------------------------
    if "num_digits" in meta:
        print("\nnum_digits")
        values, counts = np.unique(meta["num_digits"], return_counts=True)
        for v, c in zip(values, counts):
            print(f"{v} digits : {c}")

    # --------------------------------------------------
    # True digit frequencies
    # --------------------------------------------------
    if "all_digit_labels" in meta:
        print("\nActual digit occurrences")
        digits = meta["all_digit_labels"].reshape(-1)
        digits = digits[digits >= 0]  # Remove padding (-1 if present).
        counts = np.bincount(digits, minlength=10)
        for d in range(10):
            print(f"Digit {d}: {counts[d]}")

    total_labels = labels.size
    positive_labels = labels.sum()
    negative_labels = total_labels - positive_labels

    print()
    print(f"Positive labels : {positive_labels}")
    print(f"Negative labels : {negative_labels}")
    print(f"Positive ratio  : {positive_labels / total_labels:.4f}")
    print(f"Negative ratio  : {negative_labels / total_labels:.4f}")
    print()


def main() -> None:
    data_dir = Path("./data")
    splits = {
        "Train": data_dir / "train.pt",
        "Train_generated": data_dir / "train_generated.pt",
    }

    missing = [name for name, path in splits.items() if not path.exists()]
    if missing:
        print("Missing dataset files:")
        for name in missing:
            print(f"- {name}")
        return

    for name, path in splits.items():
        images, labels, meta = load_data_pt(path)
        inspect_dataset(name, images, labels, meta)


if __name__ == "__main__":
    main()
