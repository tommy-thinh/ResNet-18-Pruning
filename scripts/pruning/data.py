"""Read the original .pt data and normalize only the batches being consumed."""
from pathlib import Path

import numpy as np
from tensorflow import keras


def preprocess(images: np.ndarray) -> np.ndarray:
    if images.ndim == 3 and images.shape[1:] == (64, 64) and images.dtype == np.uint8:
        return np.repeat((images.astype(np.float32) / 255.0)[..., None], 3, axis=-1)
    if images.ndim == 4 and images.shape[1:] == (64, 64, 3) and images.dtype == np.float32:
        return images
    raise ValueError("Expected uint8 [N,64,64] or normalized float32 [N,64,64,3] images.")


def load_data_pt(path: str | Path, *, normalized: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required data split not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    images, labels = data["images"].numpy(), data["labels"].numpy()
    if images.ndim != 3 or images.shape[1:] != (64, 64) or images.dtype != np.uint8:
        raise ValueError("Expected uint8 grayscale images [N,64,64].")
    if len(images) == 0 or labels.shape != (len(images), 10) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Expected nonempty binary multi-hot labels [N,10].")
    meta = {key: value.numpy() if isinstance(value, torch.Tensor) else value
            for key, value in data.items() if key not in ("images", "labels")}
    return preprocess(images) if normalized else images, labels.astype(np.float32), meta


class Batches(keras.utils.PyDataset):
    """Shuffle small indices, not a buffer containing the full expanded dataset."""

    def __init__(self, images, labels, batch_size: int, *, shuffle: bool = False, seed: int = 42):
        # https://keras.io/api/utils/preprocessing_utils/#pydataset-class
        super().__init__(workers=2, max_queue_size=2)
        if batch_size < 1 or len(images) != len(labels) or len(images) == 0:
            raise ValueError("Use nonempty matching arrays and a positive batch size.")
        self.images, self.labels = images, labels
        self.batch_size, self.shuffle = batch_size, shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(images))
        self.on_epoch_end()

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        selected = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return preprocess(self.images[selected]), self.labels[selected]

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)
