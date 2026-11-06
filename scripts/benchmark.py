"""Measure an existing trained model once, without pruning or training."""
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
    command.add_argument("--checkpoint", type=Path, required=True)
    return command


def main() -> None:
    args = parser().parse_args()
    bench = prepare(args)
    from scripts.pruning.benchmark import evaluate_and_benchmark, runtime_info
    from scripts.pruning.model import load_model
    from scripts.pruning.reference import load_splits
    from scripts.pruning.storage import checksum

    data = load_splits(args.data_dir, ("test",))
    model = load_model(args.checkpoint)
    model.save(args.model_dir / "model.keras")
    save_result(args, {"method": "benchmark", "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checksum(args.checkpoint), "seed": args.seed,
        "runtime": runtime_info(bench), "metrics": evaluate_and_benchmark(
            model, data["test"], args.model_dir / "model.keras", bench)})


if __name__ == "__main__":
    run(main)
