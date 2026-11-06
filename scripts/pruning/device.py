"""Select a GPU before initialization and validate explicit execution placement."""
import tensorflow as tf

from .config import normalize_device


def configure_device(device: str | int = "gpu:0", memory_limit_mb: int | None = None) -> str:
    device = normalize_device(device)
    if device == "/CPU:0":
        tf.config.set_visible_devices([], "GPU")
        return "/CPU:0"
    index = int(device.rsplit(":", 1)[1])
    gpus = tf.config.list_physical_devices("GPU")
    if index >= len(gpus):
        raise RuntimeError(f"GPU {index} was requested, but TensorFlow can see {len(gpus)} GPU(s). Use --device cpu for CPU execution.")
    if memory_limit_mb is not None and memory_limit_mb < 1024:
        raise ValueError("--gpu-memory-mb must be at least 1024.")
    # Configure visibility and memory before any model variables are created.
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    selected = gpus[index]
    tf.config.set_visible_devices(selected, "GPU")
    if memory_limit_mb is None:
        tf.config.experimental.set_memory_growth(selected, True)
    else:
        tf.config.set_logical_device_configuration(selected, [
            tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)])
    # TensorFlow renumbers the only visible GPU to logical GPU 0.
    return "/GPU:0"


def require_device(device: str | int) -> None:
    device = normalize_device(device)
    if device != "/CPU:0":
        index = int(device.rsplit(":", 1)[1])
        if index >= len(tf.config.list_logical_devices("GPU")):
            raise RuntimeError(f"GPU {index} was requested, but TensorFlow cannot see that logical GPU. Use --device cpu for CPU execution.")
