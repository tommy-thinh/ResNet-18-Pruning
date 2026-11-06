"""Shared accuracy, size, sparsity and synchronized inference measurements."""
from pathlib import Path
import platform
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import BenchmarkConfig
from .data import Batches, preprocess
from .device import require_device
from .masks import eligible_layers
from .model import compile_model


def runtime_info(config: BenchmarkConfig) -> dict:
    require_device(config.device)
    hardware = platform.processor() or "CPU"
    if "GPU" in config.device.upper():
        gpus = tf.config.get_visible_devices("GPU")
        index = int(config.device.rsplit(":", 1)[1])
        hardware = tf.config.experimental.get_device_details(gpus[index]).get("device_name", "GPU")
    return {"framework": f"TensorFlow {tf.__version__}", "device": config.device, "hardware": hardware,
            "precision": "float32", "batch_size": config.batch_size}


def complexity(model: keras.Model) -> int:
    macs = 0
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            h, w, out_channels = layer.output.shape[1:]
            kh, kw, channels, _ = layer.kernel.shape
            macs += int(h * w * out_channels * kh * kw * channels)
        elif isinstance(layer, keras.layers.Dense):
            macs += int(np.prod(layer.kernel.shape))
    return macs


def timing_summary(samples: list[float], batch_size: int) -> dict:
    return {"median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
            "images_per_second": batch_size * 1000 / float(np.mean(samples))}


def evaluate_and_benchmark(model: keras.Model, test_data: tuple, checkpoint: Path,
                           config: BenchmarkConfig, eligible_names: list[str] | None = None) -> dict:
    x, y = test_data
    if len(x) < config.batch_size:
        raise ValueError("Test set is smaller than the inference batch size.")
    runtime_info(config)
    with tf.device(config.device):
        compile_model(model)
        metrics = model.evaluate(Batches(x, y, config.batch_size), return_dict=True, verbose=0)
        batch = tf.convert_to_tensor(preprocess(x[:config.batch_size]))

        @tf.function(input_signature=[tf.TensorSpec(batch.shape, tf.float32)], jit_compile=False)
        def infer(inputs):
            return model(inputs, training=False)

        for _ in range(config.warmup):
            infer(batch).numpy()
        memory_device = config.device.lstrip("/")
        try:
            tf.config.experimental.reset_memory_stats(memory_device)
        except (ValueError, tf.errors.OpError):
            pass
        samples = []
        for _ in range(config.repeats):
            start = time.perf_counter_ns()
            infer(batch).numpy()  # Materializing the output synchronizes GPU work.
            samples.append((time.perf_counter_ns() - start) / 1e6)
        try:
            peak = tf.config.experimental.get_memory_info(memory_device)["peak"] / 1024**2
        except (ValueError, tf.errors.OpError):
            peak = None
    if not all(np.isfinite(v) for v in metrics.values()):
        raise RuntimeError("Evaluation returned nonfinite metrics.")
    layers = eligible_layers(model) if eligible_names is None else [model.get_layer(n) for n in eligible_names]
    eligible = sum(int(np.prod(layer.kernel.shape)) for layer in layers)
    nonzero = sum(int(np.count_nonzero(layer.kernel.numpy())) for layer in layers)
    return {
        **{key: float(value) for key, value in metrics.items()},
        "parameters": model.count_params(), "macs": complexity(model),
        "eligible_weights": eligible, "sparsity": 1 - nonzero / eligible if eligible else None,
        "model_mb": Path(checkpoint).stat().st_size / 1024**2,
        "latency": timing_summary(samples, config.batch_size), "peak_memory_mb": peak,
    }
