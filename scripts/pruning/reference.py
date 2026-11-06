"""Shared input loading and the trained baseline's saved metrics."""
from dataclasses import asdict
import json
from pathlib import Path

from .storage import checksum


def load_splits(directory: Path, names=("train", "val", "test")) -> dict:
    from .data import load_data_pt
    return {name: load_data_pt(directory / f"{name}.pt", normalized=False)[:2] for name in names}


def baseline_result(directory: Path, config, bench, seed: int, history: dict, metrics: dict) -> dict:
    from .benchmark import runtime_info
    return {"method": "baseline", "seed": seed, "runtime": runtime_info(bench),
            "checkpoint": str(directory / "model.keras"),
            "checkpoint_sha256": checksum(directory / "model.keras"),
            "initialization": str(directory / "initial.keras"),
            "initialization_sha256": checksum(directory / "initial.keras"),
            "training": {"config": asdict(config), **history}, "metrics": metrics}


def load_baseline(path: Path, checkpoint: Path) -> dict:
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("status") != "completed" or "metrics" not in reference:
        raise ValueError("Baseline JSON must contain completed measurements of the trained model.")
    if reference.get("checkpoint_sha256") != checksum(checkpoint):
        raise ValueError("Checkpoint does not match the saved baseline metrics.")
    return reference


def pruning_result(method: str, args, bench, details: dict, reference: dict) -> dict:
    from .benchmark import runtime_info
    return {"method": method, "status": "incomplete", "checkpoint": str(args.checkpoint),
            "seed": args.seed, "runtime": runtime_info(bench), "pruning": details,
            "baseline": str(args.baseline), "dense": reference["metrics"],
            "dense_runtime": reference["runtime"]}
