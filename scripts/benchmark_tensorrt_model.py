"""Benchmark one trained dense, unstructured or channel-pruned model with TensorRT FP16."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.cli import add_common_arguments, add_tensorrt_arguments, prepare, run, save_result


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(command)
    add_tensorrt_arguments(command)
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Export this trained .keras checkpoint first")
    source.add_argument("--onnx", type=Path, help="Reuse an existing ONNX export")
    return command


def main():
    args = parser().parse_args()
    if args.device == "/CPU:0":
        raise ValueError("TensorRT requires a GPU device.")
    bench = prepare(args)
    from scripts.pruning.inference import benchmark_onnx
    from scripts.pruning.tensorrt_runtime import select_cuda_device

    source = args.onnx
    if args.checkpoint:
        source = args.model_dir / "model.onnx"
        # tf2onnx creates separate TF1 sessions; isolate offline export from all GPUs.
        with tempfile.TemporaryDirectory(prefix="cnn-onnx-", dir=args.model_dir.parent) as scratch:
            subprocess.run([sys.executable, str(Path(__file__).with_name("export_onnx.py")),
                "--checkpoint", str(args.checkpoint), "--model-dir", str(args.model_dir),
                "--data-dir", str(args.data_dir), "--device", "cpu",
                "--output", str(Path(scratch) / "export.json")], check=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    select_cuda_device(args.device)
    result = benchmark_onnx(source, args.data_dir / "test.pt", args.model_dir, bench,
                            cuda_graph=args.cuda_graph, profile=args.profile)
    if args.checkpoint:
        result["checkpoint"] = str(args.checkpoint)
    save_result(args, result)


if __name__ == "__main__":
    run(main)
