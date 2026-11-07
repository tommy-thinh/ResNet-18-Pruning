"""Common arguments and runtime setup; experiment logic lives in each script."""
import argparse
import os
from pathlib import Path
import sys

from .config import normalize_device


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True, help="One result JSON")
    parser.add_argument("--model-dir", type=Path, help="Default: models/<output filename stem>")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=normalize_device, default="/GPU:0",
                        help="GPU index (0, 1, ...), gpu:N, /GPU:N, or cpu; default: 0")
    parser.add_argument("--gpu-memory-mb", type=int,
                        help="Optional TensorFlow GPU memory cap in MB; default: grow as needed")
    parser.add_argument("--batch-size", type=int, default=64)


def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)


def add_training_arguments(parser: argparse.ArgumentParser, *, epochs=100, learning_rate=0.01) -> None:
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--learning-rate", type=float, default=learning_rate)


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("results/baseline.json"),
                        help="Saved metrics for the trained dense model")


def retained_density(value: str) -> float:
    density = float(value)
    if not 0 < density < 1:
        raise argparse.ArgumentTypeError(
            "Density must be between 0 and 1 (exclusive). Use the trained baseline metrics for density 1.")
    return density


def prepare(args):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import tensorflow as tf
    from .config import BenchmarkConfig
    from .device import configure_device
    from .model import set_seed
    from .storage import prepare_output

    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    device = configure_device(args.device, args.gpu_memory_mb)
    set_seed(args.seed)
    args.model_dir = args.model_dir or Path("models") / args.output.stem
    prepare_output(args.output, args.model_dir)
    return BenchmarkConfig(batch_size=args.batch_size, warmup=getattr(args, "warmup", 10),
                           repeats=getattr(args, "repeats", 50),
                           device=device)


def training_config(args, *, device: str | None = None):
    from .config import TrainingConfig
    return TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                          device=args.device if device is None else device)


def save_result(args, result: dict) -> None:
    from .storage import write_json
    result["status"] = "completed"
    write_json(args.output, result)
    print(f"Result: {args.output}")


def run(main) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
