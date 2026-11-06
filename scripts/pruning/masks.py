"""Persistent magnitude and Ampere-compatible reduction-axis 2:4 masks."""
import base64
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def eligible_layers(model: keras.Model, *, include_output: bool = False,
                    conv_only: bool = False, exclude: tuple[str, ...] = ()) -> list:
    types = (keras.layers.Conv2D,) if conv_only else (keras.layers.Conv2D, keras.layers.Dense)
    return [layer for layer in model.layers if isinstance(layer, types)
            and (include_output or layer.name != "predictions") and layer.name not in exclude]


def magnitude_masks(weights: dict[str, np.ndarray], density: float,
                    previous: dict[str, np.ndarray] | None = None,
                    layerwise: bool = False) -> dict[str, np.ndarray]:
    """Keep round(N*density) entries, breaking magnitude ties by stable index.

    With previous masks, density is still relative to the original eligible count.
    Removed entries can never return, even if survivors have become exactly zero.
    """
    if not np.isfinite(density) or not 0 < density <= 1 or not weights:
        raise ValueError("Expected nonempty kernels and density in (0,1].")
    if layerwise:
        return {name: magnitude_masks({name: value}, density,
                {name: previous[name]} if previous is not None else None)[name]
                for name, value in weights.items()}
    names = list(weights)
    if any(not np.isfinite(value).all() for value in weights.values()):
        raise ValueError("Pruning requires finite weights.")
    if previous is not None:
        validate_masks(weights, previous)
    sizes = [weights[name].size for name in names]
    active = np.concatenate([(np.ones_like(weights[name], dtype=bool) if previous is None
                              else previous[name].astype(bool)).ravel() for name in names])
    keep = int(np.floor(sum(sizes) * density + 0.5))
    candidates = np.flatnonzero(active)
    if keep > len(candidates):
        raise ValueError("Target density would regrow previously removed weights.")
    values = np.concatenate([np.abs(weights[name]).ravel() for name in names])
    order = np.argsort(-values[candidates], kind="stable")
    mask = np.zeros(len(values), dtype=bool)
    mask[candidates[order[:keep]]] = True
    pieces = np.split(mask, np.cumsum(sizes)[:-1])
    return {name: part.reshape(weights[name].shape) for name, part in zip(names, pieces)}


def validate_masks(weights: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> None:
    if weights.keys() != masks.keys():
        raise ValueError("Mask keys must exactly match eligible layer names.")
    for name, weight in weights.items():
        if masks[name].shape != weight.shape or not np.isin(masks[name], (0, 1)).all():
            raise ValueError(f"Invalid mask for {name}.")


def mask_2of4(weight: np.ndarray) -> np.ndarray:
    # Keras Conv: [R,S,C,K] => group C, Dense: [C,K] => group C.
    # https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/io-formats-sparsity.html
    axis = 2 if weight.ndim == 4 else 0 if weight.ndim == 2 else None
    if axis is None or weight.shape[axis] % 4:
        raise ValueError("2:4 requires a Conv2D/Dense kernel with reduction dimension divisible by four.")
    if not np.isfinite(weight).all():
        raise ValueError("Nonfinite weights cannot be pruned.")
    moved = np.moveaxis(weight, axis, -1)
    groups = moved.reshape(*moved.shape[:-1], -1, 4)
    order = np.argsort(-np.abs(groups), axis=-1, kind="stable")
    mask = np.zeros_like(groups, dtype=bool)
    np.put_along_axis(mask, order[..., :2], True, axis=-1)
    return np.moveaxis(mask.reshape(moved.shape), -1, axis)


def valid_2of4(weight: np.ndarray) -> bool:
    axis = 2 if weight.ndim == 4 else 0 if weight.ndim == 2 else None
    if axis is None or weight.shape[axis] % 4:
        return False
    moved = np.moveaxis(weight, axis, -1)
    return bool((np.count_nonzero(moved.reshape(-1, 4), axis=-1) <= 2).all())


def semi_structured_masks(model: keras.Model, **eligibility) -> tuple[dict, dict]:
    masks, excluded = {}, {}
    for layer in eligible_layers(model, **eligibility):
        w = layer.kernel.numpy()
        axis = 2 if w.ndim == 4 else 0
        if w.shape[axis] % 4:
            excluded[layer.name] = f"reduction dimension {w.shape[axis]} is not divisible by 4"
        else:
            masks[layer.name] = mask_2of4(w)
    if not masks:
        raise ValueError("No layers eligible for 2:4 pruning.")
    return masks, excluded


@keras.utils.register_keras_serializable(package="pruning_study")
class FixedMask(keras.constraints.Constraint):
    """Serializable binary projection after every optimizer update."""

    def __init__(self, packed: str, shape: list[int]):
        self.packed = packed
        self.shape = tuple(shape)
        bits = np.unpackbits(np.frombuffer(base64.b64decode(packed), dtype=np.uint8))
        self.mask = bits[:int(np.prod(shape))].reshape(shape).astype(bool)

    @classmethod
    def from_mask(cls, mask: np.ndarray):
        packed = base64.b64encode(np.packbits(mask.ravel()).tobytes()).decode("ascii")
        return cls(packed, list(mask.shape))

    def __call__(self, weight):
        return weight * tf.cast(self.mask, weight.dtype)

    def get_config(self) -> dict:
        return {"packed": self.packed, "shape": list(self.shape)}


@keras.utils.register_keras_serializable(package="pruning_study")
class MaskedSGD(keras.optimizers.SGD):
    """Mask gradients as well as updates, keeping momentum/EMA slots sparse too."""

    def apply_gradients(self, grads_and_vars):
        pairs = []
        for gradient, variable in grads_and_vars:
            constraint = variable.constraint
            if gradient is not None and isinstance(constraint, FixedMask):
                gradient = constraint(gradient)
            pairs.append((gradient, variable))
        return super().apply_gradients(pairs)


def apply_masks(model: keras.Model, masks: dict[str, np.ndarray]) -> None:
    for name, mask in masks.items():
        layer = model.get_layer(name)
        validate_masks({name: layer.kernel.numpy()}, {name: mask})
        constraint = FixedMask.from_mask(mask)
        layer.kernel_constraint = constraint
        layer.kernel.constraint = constraint
        layer.kernel.assign(constraint(layer.kernel))


def assert_masks(model: keras.Model, masks: dict[str, np.ndarray]) -> None:
    for name, mask in masks.items():
        if np.any(model.get_layer(name).kernel.numpy()[~mask.astype(bool)] != 0):
            raise AssertionError(f"Pruned weights regrew in {name}")


def model_masks(model: keras.Model) -> dict[str, np.ndarray]:
    return {layer.name: layer.kernel.constraint.mask for layer in eligible_layers(model, include_output=True)
            if isinstance(layer.kernel.constraint, FixedMask)}


def save_masks(path: str | Path, masks: dict) -> None:
    np.savez_compressed(path, **masks)


def load_masks(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
