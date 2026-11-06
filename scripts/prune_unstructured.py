"""Globally magnitude-prune and fine-tune one sparse model."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.cli import (add_common_arguments, add_benchmark_arguments, add_source_arguments,
                                 add_training_arguments, prepare, retained_density, run, save_result, training_config)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(command)
    add_source_arguments(command)
    add_benchmark_arguments(command)
    add_training_arguments(command)
    command.add_argument("--density", type=retained_density, required=True)
    return command


def main() -> None:
    args = parser().parse_args()
    bench = prepare(args)
    from scripts.pruning.masks import apply_masks, eligible_layers, magnitude_masks, save_masks
    from scripts.pruning.model import load_model
    from scripts.pruning.reference import load_baseline, load_splits, pruning_result
    from scripts.pruning.training import finetune_and_measure

    reference = load_baseline(args.baseline, args.checkpoint)
    data = load_splits(args.data_dir)
    model = load_model(args.checkpoint)
    masks = magnitude_masks({layer.name: layer.kernel.numpy() for layer in eligible_layers(model)}, args.density)
    apply_masks(model, masks)
    save_masks(args.model_dir / "masks.npz", masks)
    result = pruning_result("unstructured", args, bench, {"density": args.density}, reference)
    _, result = finetune_and_measure(model, data, args, training_config(args), bench, result, masks)
    save_result(args, result)


if __name__ == "__main__":
    run(main)
