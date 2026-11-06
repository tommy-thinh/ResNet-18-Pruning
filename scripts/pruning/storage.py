"""Small JSON results and checkpoint checksums, without a run registry."""
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def encode(item):
        if is_dataclass(item):
            return asdict(item)
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot encode {type(item).__name__}")

    path.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + "\n", encoding="utf-8")


def checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(output: Path, model_dir: Path) -> None:
    if output.suffix.lower() != ".json":
        raise ValueError("--output must name a .json file.")
    if output.exists():
        raise FileExistsError(f"Result already exists: {output}. Choose another --output path.")
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(f"Checkpoint directory is not empty: {model_dir}. Choose another --model-dir.")
    model_dir.mkdir(parents=True, exist_ok=True)
