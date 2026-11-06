"""Compare dense and sparse TensorRT engines, including a same-weights control."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.cli import add_common_arguments, add_benchmark_arguments, prepare, run, save_result


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(command)
    add_benchmark_arguments(command)
    command.add_argument("--dense-onnx", type=Path, required=True)
    command.add_argument("--sparse-onnx", type=Path, required=True)
    return command


def main() -> None:
    args = parser().parse_args()
    if args.device == "cpu":
        raise ValueError("TensorRT requires --device gpu:0.")
    bench = prepare(args)
    from scripts.pruning.inference import tensorrt_comparison
    result = tensorrt_comparison(args.dense_onnx, args.sparse_onnx, args.data_dir / "test.pt", args.model_dir, bench)
    save_result(args, result)


if __name__ == "__main__":
    run(main)
