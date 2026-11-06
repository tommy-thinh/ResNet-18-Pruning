"""Export a trained model and verify ONNX parity and sparsity."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.cli import add_common_arguments, prepare, run, save_result


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(command)
    command.add_argument("--checkpoint", type=Path, required=True)
    return command


def main() -> None:
    args = parser().parse_args()
    prepare(args)
    from scripts.pruning.inference import export_onnx
    result = export_onnx(args.checkpoint, args.model_dir / "model.onnx", args.data_dir / "test.pt")
    save_result(args, result)


if __name__ == "__main__":
    run(main)
