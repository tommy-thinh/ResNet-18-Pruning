"""Select GPU 0 before initialization and validate explicit execution placement."""
import tensorflow as tf


def configure_device(device: str = "gpu:0", memory_limit_mb: int = 4096) -> str:
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return "/CPU:0"
    if device not in ("gpu", "gpu:0"):
        raise ValueError("Use gpu:0 or cpu.")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU 0 was requested, but TensorFlow cannot see a GPU. Use --device cpu for CPU execution.")
    if memory_limit_mb < 1024:
        raise ValueError("--gpu-memory-mb must be at least 1024.")
    # Configure visibility and memory before any model variables are created.
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    tf.config.set_visible_devices(gpus[0], "GPU")
    tf.config.set_logical_device_configuration(gpus[0], [
        tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)])
    return "/GPU:0"


def require_device(device: str) -> None:
    if device not in ("/CPU:0", "/GPU:0"):
        raise ValueError("Only /GPU:0 and /CPU:0 are supported.")
    if device == "/GPU:0" and not tf.config.list_logical_devices("GPU"):
        raise RuntimeError("GPU 0 was requested, but TensorFlow cannot see a GPU. Use --device cpu for CPU execution.")
