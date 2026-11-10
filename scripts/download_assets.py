"""Download the published checkpoints/data without overwriting local artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "fucthin/ResNet-18-Pruning-Datasets-and-Models"


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(repo_id: str, repo_type: str, revision: str, destination: Path, files: dict) -> None:
    from huggingface_hub import snapshot_download

    destination = destination.resolve()
    pending = []
    for name, metadata in files.items():
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Artifact path escapes destination: {name}")
        if target.exists():
            if checksum(target) != metadata["sha256"]:
                raise FileExistsError(f"Refusing to overwrite a different local file: {target}")
        else:
            pending.append(name)
    if not pending:
        print(f"Verified existing {destination.name} files")
        return
    destination.mkdir(parents=True, exist_ok=True)
    # https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder
    with tempfile.TemporaryDirectory(prefix=".download-", dir=destination) as scratch:
        snapshot_download(repo_id=repo_id, repo_type=repo_type, revision=revision,
                          allow_patterns=pending, local_dir=scratch)
        for name in pending:
            source = Path(scratch) / name
            if not source.is_file() or checksum(source) != files[name]["sha256"]:
                raise ValueError(f"Missing or mismatched published artifact: {name}")
        for name in pending:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with (Path(scratch) / name).open("rb") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    print(f"Downloaded and verified {len(pending)} {destination.name} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repository: owner/name")
    parser.add_argument("--revision", default="main", help="Commit hash or branch")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "artifacts.json").read_text(encoding="utf-8"))
    files = {f"{group}/{name}": metadata
             for group in ("models", "data") for name, metadata in manifest[group].items()}
    download(args.repo, "dataset", args.revision, ROOT, files)


if __name__ == "__main__":
    main()
