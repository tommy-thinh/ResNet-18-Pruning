"""Apply hardware-aligned 2:4 masks and fine-tune one model."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.cli import (add_common_arguments, add_benchmark_arguments, add_source_arguments,
                                 add_training_arguments, prepare, run, save_result, training_config)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(command)
    add_source_arguments(command)
    add_benchmark_arguments(command)
    add_training_arguments(command)
    return command


def main() -> None:
    args = parser().parse_args()
    bench = prepare(args)
    config = training_config(args, device=bench.device)
    from scripts.pruning.masks import apply_masks, save_masks, semi_structured_masks, valid_2of4
    from scripts.pruning.model import load_model
    from scripts.pruning.reference import load_baseline, load_splits, pruning_result
    from scripts.pruning.training import finetune_and_measure

    reference = load_baseline(args.baseline, args.checkpoint)
    data = load_splits(args.data_dir)
    model = load_model(args.checkpoint)
    masks, excluded = semi_structured_masks(model)
    apply_masks(model, masks)
    save_masks(args.model_dir / "masks.npz", masks)
    details = {"density": 0.5, "excluded": excluded}
    result = pruning_result("semi24", args, bench, details, reference)
    model, result = finetune_and_measure(model, data, args, config, bench, result, masks)
    details["pattern_valid"] = all(valid_2of4(model.get_layer(name).kernel.numpy()) for name in masks)
    if not details["pattern_valid"]:
        raise AssertionError("Fine-tuning changed the 2:4 pattern.")
    save_result(args, result)


if __name__ == "__main__":
    run(main)
