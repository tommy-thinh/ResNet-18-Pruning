"""Validation-selected checkpoints and fresh optimizer state for every training run."""
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import TrainingConfig
from .data import Batches
from .device import require_device
from .masks import apply_masks, assert_masks
from .model import compile_model, load_model, set_seed


class MaskGuard(keras.callbacks.Callback):
    def __init__(self, masks: dict):
        super().__init__()
        self.masks = masks

    def on_epoch_end(self, epoch, logs=None):
        assert_masks(self.model, self.masks)

    def on_train_end(self, logs=None):
        apply_masks(self.model, self.masks)
        assert_masks(self.model, self.masks)


def training_callbacks(directory: Path, config: TrainingConfig, masks: dict | None = None) -> list:
    directory.mkdir(parents=True, exist_ok=True)
    callbacks = [MaskGuard(masks)] if masks else []
    return callbacks + [
        keras.callbacks.ModelCheckpoint(str(directory / "best.keras"),
            monitor="val_exact_match_accuracy", mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_exact_match_accuracy", mode="max",
            patience=config.early_stopping_patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=config.lr_factor,
            patience=config.lr_patience, min_lr=config.min_lr),
        keras.callbacks.TerminateOnNaN(),
    ]


def save_initialization(model: keras.Model, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "initial.keras"
    if path.exists():
        raise FileExistsError(f"Initialization already exists: {path}")
    if getattr(model, "optimizer", None) is not None:
        raise ValueError("Save initialization before compiling or training.")
    model.save(path)
    return path


def train(model: keras.Model, train_data: tuple, val_data: tuple, config: TrainingConfig,
          directory: Path, *, seed: int, masks: dict | None = None) -> tuple[keras.Model, dict]:
    require_device(config.device)
    set_seed(seed)

    def clone_layer(layer):
        settings = layer.get_config()
        if isinstance(layer, keras.layers.Dropout):
            settings["seed"] = seed
        return layer.__class__.from_config(settings)

    start = time.perf_counter()
    # Scope variable/optimizer creation and both train/validation steps together.
    # https://www.tensorflow.org/guide/gpu#manual_device_placement
    with tf.device(config.device):
        fresh = keras.models.clone_model(model, clone_function=clone_layer)
        fresh.set_weights(model.get_weights())
        model = fresh
        if masks:
            apply_masks(model, masks)
        compile_model(model, config)
        history = model.fit(Batches(*train_data, config.batch_size, shuffle=True, seed=seed),
            validation_data=Batches(*val_data, config.batch_size), epochs=config.epochs,
            callbacks=training_callbacks(directory, config, masks), verbose=2, shuffle=False)
    values = {key: [float(v) for v in history.history[key]] for key in (
        "loss", "val_loss", "exact_match_accuracy", "val_exact_match_accuracy", "learning_rate")
        if key in history.history}
    if not values.get("val_exact_match_accuracy") or not all(
        np.isfinite(v) for series in values.values() for v in series
    ):
        raise RuntimeError("Training did not produce finite train/validation metrics.")
    best_epoch = int(np.argmax(values["val_exact_match_accuracy"])) + 1
    record = {"epochs_run": len(history.epoch), "best_epoch": best_epoch,
              "best_iteration": best_epoch * int(np.ceil(len(train_data[0]) / config.batch_size)),
              "seconds": time.perf_counter() - start, "history": values}
    # Reload the raw best epoch after fit() finalizes EMA; do not evaluate the
    # unvalidated final EMA average. Inference-only saves make sizes comparable.
    with tf.device(config.device):
        best = load_model(directory / "best.keras")
        if masks:
            assert_masks(best, masks)
        best.save(directory / "model.keras")
    return best, record


def reset_to_initialization(model: keras.Model, initial: keras.Model, masks: dict) -> None:
    model.set_weights(initial.get_weights())
    apply_masks(model, masks)


def random_control(initial: keras.Model, masks: dict, seed: int) -> keras.Model:
    set_seed(seed)
    control = keras.models.clone_model(initial)
    for layer in control.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            source = initial.get_layer(layer.name)
            layer.moving_mean.assign(source.moving_mean)
            layer.moving_variance.assign(source.moving_variance)
    apply_masks(control, masks)
    return control


def finetune_and_measure(model, data: dict, args, config: TrainingConfig, bench,
                         result: dict, masks: dict | None = None):
    """The common pre/post measurement steps, independent of how a model was pruned."""
    from .benchmark import evaluate_and_benchmark
    from .storage import write_json

    names = list(masks) if masks else None
    model.save(args.model_dir / "pruned.keras")
    result["before"] = evaluate_and_benchmark(model, data["test"], args.model_dir / "pruned.keras", bench, names)
    write_json(args.output, result)
    model, history = train(model, data["train"], data["val"], config, args.model_dir,
                           seed=args.seed, masks=masks)
    assert_masks(model, masks or {})
    result["training"] = {"config": asdict(config), **history}
    result["after"] = evaluate_and_benchmark(model, data["test"], args.model_dir / "model.keras", bench, names)
    return model, result
