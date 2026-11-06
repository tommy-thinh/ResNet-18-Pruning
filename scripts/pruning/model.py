"""The notebook's single-output ResNet-18 and compact variants."""

import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


@keras.utils.register_keras_serializable(package="pruning_study")
def exact_match_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # Per-example values let Keras weight partial batches correctly.
    predicted = tf.cast(y_pred >= 0.5, y_true.dtype)
    return tf.cast(tf.reduce_all(tf.equal(y_true, predicted), axis=-1), tf.float32)


def basic_block(x, filters: int, stride: int = 1, name: str | None = None):
    shortcut = x
    y = layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv1"
    )(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(
        y
    )
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"{name}_proj_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    y = layers.Add(name=f"{name}_add")([shortcut, y])
    return layers.ReLU(name=f"{name}_out_relu")(y)


def build_resnet18_model(
    input_shape=(64, 64, 3), widths=(64, 128, 256, 512), dropout_seed=42
) -> keras.Model:
    if len(widths) != 4 or any(int(n) != n or n < 1 for n in widths):
        raise ValueError("Four positive integer stage widths are required.")
    inputs = keras.Input(shape=input_shape, name="images")
    x = layers.Conv2D(
        widths[0], 7, strides=2, padding="same", use_bias=False, name="conv1"
    )(inputs)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    x = layers.ReLU(name="conv1_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)
    for stage, width in enumerate(widths, start=2):
        for block in (1, 2):
            stride = 2 if stage > 2 and block == 1 else 1
            x = basic_block(x, width, stride, name=f"conv{stage}_block{block}")
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, seed=dropout_seed, name="dropout")(x)
    outputs = layers.Dense(10, activation="sigmoid", name="predictions")(x)
    return keras.Model(inputs, outputs, name="multi_label_resnet18")


def compile_model(model: keras.Model, config=None) -> None:
    """Configure evaluation, or training when an explicit training config is supplied."""
    # Evaluation needs loss/metrics, not a training optimizer or training defaults.
    # https://github.com/keras-team/keras/blob/v3.12.0/keras/src/trainers/trainer.py
    optimizer = None
    if config is not None:
        from .masks import MaskedSGD, model_masks

        # EMA is finalized by fit(); always reload the best epoch afterward.
        # https://keras.io/api/optimizers/sgd/
        optimizer_class = MaskedSGD if model_masks(model) else keras.optimizers.SGD
        optimizer = optimizer_class(
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            use_ema=config.use_ema,
        )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            exact_match_accuracy,
        ],
        jit_compile=False,
    )


def load_model(path) -> keras.Model:
    # Import registration for serialized mask constraints as well as metrics.
    from . import masks  # noqa: F401

    return keras.models.load_model(path, compile=False)
