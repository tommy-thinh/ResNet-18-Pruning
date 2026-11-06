"""Remove residual-stage channels, build a compact model and fine-tune it."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.cli import (add_common_arguments, add_benchmark_arguments, add_source_arguments,
                                 add_training_arguments, prepare, run, save_result, training_config)


def removal_fraction(value: str) -> float:
    fraction = float(value)
    if not 0 < fraction < 1:
        raise argparse.ArgumentTypeError("Channel removal must be between 0 and 1 (exclusive).")
    return fraction


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(command)
    add_source_arguments(command)
    add_benchmark_arguments(command)
    add_training_arguments(command)
    command.add_argument("--remove", type=removal_fraction, required=True, help="Channel removal fraction, e.g. 0.2")
    return command


def main() -> None:
    args = parser().parse_args()
    bench = prepare(args)
    config = training_config(args, device=bench.device)
    from scripts.pruning.channels import prune_channels
    from scripts.pruning.model import load_model
    from scripts.pruning.reference import load_baseline, load_splits, pruning_result
    from scripts.pruning.training import finetune_and_measure

    reference = load_baseline(args.baseline, args.checkpoint)
    data = load_splits(args.data_dir)
    model, plan = prune_channels(load_model(args.checkpoint), args.remove)
    details = {"channel_removal": args.remove, "widths": plan["compact_widths"]}
    result = pruning_result("channels", args, bench, details, reference)
    _, result = finetune_and_measure(model, data, args, config, bench, result)
    save_result(args, result)


if __name__ == "__main__":
    run(main)
