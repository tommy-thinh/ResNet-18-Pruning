"""Measure an existing TensorRT engine without exporting, rebuilding or training."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.cli import add_common_arguments, add_tensorrt_arguments, prepare, run, save_result


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(command)
    add_tensorrt_arguments(command)
    command.add_argument("--engine", type=Path, required=True)
    return command


def main():
    args = parser().parse_args()
    if args.device == "/CPU:0":
        raise ValueError("TensorRT requires a GPU device.")
    bench = prepare(args)
    from scripts.pruning.inference import benchmark_saved_engine
    from scripts.pruning.tensorrt_runtime import select_cuda_device

    select_cuda_device(args.device)
    result = benchmark_saved_engine(args.engine, args.data_dir / "test.pt", bench,
                                    cuda_graph=args.cuda_graph, profile=args.profile)
    save_result(args, result)


if __name__ == "__main__":
    run(main)
