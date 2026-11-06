"""Train and measure the reference dense ResNet-18."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.cli import (add_common_arguments, add_benchmark_arguments,
                                 add_training_arguments, prepare, run, save_result, training_config)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(command)
    add_benchmark_arguments(command)
    add_training_arguments(command, epochs=200, learning_rate=0.1)
    return command


def main() -> None:
    args = parser().parse_args()
    bench = prepare(args)
    from scripts.pruning.benchmark import evaluate_and_benchmark
    from scripts.pruning.model import build_resnet18_model
    from scripts.pruning.reference import baseline_result, load_splits
    from scripts.pruning.training import save_initialization, train

    data = load_splits(args.data_dir)
    config = training_config(args, device=bench.device)
    model = build_resnet18_model(dropout_seed=args.seed)
    save_initialization(model, args.model_dir)
    model, history = train(model, data["train"], data["val"], config, args.model_dir, seed=args.seed)
    metrics = evaluate_and_benchmark(model, data["test"], args.model_dir / "model.keras", bench)
    save_result(args, baseline_result(args.model_dir, config, bench, args.seed, history, metrics))


if __name__ == "__main__":
    run(main)
