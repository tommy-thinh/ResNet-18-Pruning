"""Discover an iterative lottery ticket and train its random-initialization control."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.cli import (add_common_arguments, add_benchmark_arguments, add_source_arguments,
                                 add_training_arguments, prepare, run, save_result)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(command)
    add_source_arguments(command)
    add_benchmark_arguments(command)
    add_training_arguments(command, epochs=None, learning_rate=None)
    command.add_argument("--initialization", type=Path, required=True)
    command.add_argument("--rounds", type=int, required=True)
    command.add_argument("--prune-rate", type=float, default=0.2)
    return command


def discover(args, bench) -> dict:
    import numpy as np
    from tensorflow import keras
    from scripts.pruning.benchmark import evaluate_and_benchmark, runtime_info
    from scripts.pruning.config import TrainingConfig
    from scripts.pruning.masks import eligible_layers, magnitude_masks, save_masks
    from scripts.pruning.model import load_model, set_seed
    from scripts.pruning.reference import load_baseline, load_splits
    from scripts.pruning.storage import checksum, write_json
    from scripts.pruning.training import random_control, reset_to_initialization, train

    reference = load_baseline(args.baseline, args.checkpoint)
    if reference.get("method") != "baseline":
        raise ValueError("Lottery Ticket comparisons require the matching training baseline JSON.")
    if reference["seed"] != args.seed or reference["initialization_sha256"] != checksum(args.initialization):
        raise ValueError("Initialization does not match the baseline's checksum/seed.")
    config = replace(TrainingConfig(**reference["training"]["config"]), device=bench.device)
    if ((args.epochs is not None and args.epochs != config.epochs)
            or (args.learning_rate is not None and args.learning_rate != config.learning_rate)):
        raise ValueError("Lottery Ticket training must use the baseline's training budget and learning rate.")
    if bench.batch_size != config.batch_size:
        raise ValueError("Lottery Ticket batch size must match the baseline.")
    if args.rounds < 1 or not 0 < args.prune_rate < 1:
        raise ValueError("Use positive --rounds and a --prune-rate in (0,1).")
    data = load_splits(args.data_dir)
    initial, current = load_model(args.initialization), load_model(args.checkpoint)
    names = [layer.name for layer in eligible_layers(current, conv_only=True)
             if not layer.name.endswith("_proj_conv")]
    masks = {name: np.ones(current.get_layer(name).kernel.shape, dtype=bool) for name in names}
    result = {"method": "lth", "status": "incomplete", "seed": args.seed,
              "runtime": runtime_info(bench), "checkpoint": str(args.checkpoint),
              "baseline": str(args.baseline), "dense": reference["metrics"],
              "dense_runtime": reference["runtime"], "prune_rate": args.prune_rate,
              "training_config": asdict(config), "rounds": []}
    for index in range(1, args.rounds + 1):
        directory = args.model_dir / f"round-{index}"
        directory.mkdir()
        weights = {name: current.get_layer(name).kernel.numpy() for name in names}
        density = sum(int(mask.sum()) for mask in masks.values()) / sum(mask.size for mask in masks.values())
        masks = magnitude_masks(weights, density * (1 - args.prune_rate), previous=masks)
        del current, weights
        keras.backend.clear_session()
        set_seed(args.seed)
        current = keras.models.clone_model(initial)
        reset_to_initialization(current, initial, masks)
        save_masks(directory / "masks.npz", masks)
        current.save(directory / "reset.keras")
        current, history = train(current, data["train"], data["val"], config, directory,
                                 seed=args.seed, masks=masks)
        actual_density = sum(int(mask.sum()) for mask in masks.values()) / sum(mask.size for mask in masks.values())
        result["rounds"].append({"round": index, "density": actual_density, "training": history})
        write_json(args.output, result)
    result["original"] = evaluate_and_benchmark(current, data["test"], directory / "model.keras", bench, names)
    control_seed = args.seed + 10000
    control = random_control(initial, masks, control_seed)
    control, history = train(control, data["train"], data["val"], config, args.model_dir / "random",
                             seed=args.seed, masks=masks)
    result["random_initialization_seed"] = control_seed
    result["random_training"] = history
    result["random"] = evaluate_and_benchmark(control, data["test"], args.model_dir / "random" / "model.keras", bench, names)
    return result


def main() -> None:
    args = parser().parse_args()
    bench = prepare(args)
    save_result(args, discover(args, bench))


if __name__ == "__main__":
    run(main)
