"""Download the published checkpoints/data without overwriting local artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--model-repo", help="Hugging Face model repository: owner/name")
    parser.add_argument("--dataset-repo", help="Hugging Face dataset repository: owner/name")
    parser.add_argument("--model-revision", default="main", help="Commit hash or branch")
    parser.add_argument("--dataset-revision", default="main", help="Commit hash or branch")
    args = parser.parse_args()
    if not (args.model_repo or args.dataset_repo):
        parser.error("Supply --model-repo and/or --dataset-repo")
    manifest = json.loads((ROOT / "artifacts.json").read_text(encoding="utf-8"))
    if args.model_repo:
        download(args.model_repo, "model", args.model_revision, ROOT / "models", manifest["models"])
    if args.dataset_repo:
        download(args.dataset_repo, "dataset", args.dataset_revision, ROOT / "data", manifest["data"])


if __name__ == "__main__":
    main()
