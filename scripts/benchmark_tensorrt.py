"""Compare dense and sparse TensorRT engines, including a same-weights control."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.cli import add_common_arguments, add_tensorrt_arguments, prepare, run, save_result


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(command)
    add_tensorrt_arguments(command)
    command.add_argument("--dense-onnx", type=Path, required=True)
    command.add_argument("--sparse-onnx", type=Path, required=True)
    return command


def main() -> None:
    args = parser().parse_args()
    if args.device == "/CPU:0":
        raise ValueError("TensorRT requires a GPU device.")
    bench = prepare(args)
    from scripts.pruning.inference import tensorrt_comparison
    from scripts.pruning.tensorrt_runtime import select_cuda_device

    # TensorFlow visibility does not select the CUDA device used by TensorRT.
    select_cuda_device(args.device)
    result = tensorrt_comparison(args.dense_onnx, args.sparse_onnx, args.data_dir / "test.pt", args.model_dir, bench,
                                cuda_graph=args.cuda_graph, profile=args.profile)
    save_result(args, result)


if __name__ == "__main__":
    run(main)
