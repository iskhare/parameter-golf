#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "iskhare/parameter-golf-artifacts"
DEFAULT_REPO_TYPE = "dataset"


PRESETS: dict[str, list[tuple[str, str]]] = {
    "basic-smoke": [
        ("data/tokenizers/fineweb_4096_bpe.model", "tokenizers/fineweb_4096_bpe.model"),
        ("data/tokenizers/fineweb_4096_bpe.vocab", "tokenizers/fineweb_4096_bpe.vocab"),
        ("data/tokenizers/fineweb_8192_bpe.model", "tokenizers/fineweb_8192_bpe.model"),
        ("data/tokenizers/fineweb_8192_bpe.vocab", "tokenizers/fineweb_8192_bpe.vocab"),
        ("logs/experiments/results.jsonl", "runs/smoke/results.jsonl"),
        ("logs/experiments/sp4096_mlp4_qk525_wd_seed1337_iter20.txt", "runs/smoke/sp4096_mlp4_qk525_wd_seed1337_iter20.txt"),
        ("logs/experiments/sp4096_mlp4_qk525_wd_seed1337_iter20_mlx_model.int8.ptz", "best/sp4096_mlp4_qk525_wd_seed1337_iter20_mlx_model.int8.ptz"),
        ("TODO.md", "notes/TODO.md"),
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload or download small experiment artifacts to Hugging Face.")
    parser.add_argument("mode", choices=("upload", "download"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="basic-smoke")
    parser.add_argument("--local-path", action="append", default=[], help="Extra local file to upload.")
    parser.add_argument("--remote-path", action="append", default=[], help="Explicit remote path(s) matching --local-path order.")
    parser.add_argument("--download-dir", default="hf_artifacts")
    parser.add_argument("--list-only", action="store_true")
    return parser


def collect_entries(args: argparse.Namespace) -> list[tuple[Path, str]]:
    entries = [(Path(local), remote) for local, remote in PRESETS[args.preset]]
    if args.local_path:
        if args.remote_path and len(args.local_path) != len(args.remote_path):
            raise ValueError("--remote-path count must match --local-path count")
        for idx, local in enumerate(args.local_path):
            remote = args.remote_path[idx] if args.remote_path else Path(local).name
            entries.append((Path(local), remote))
    return entries


def upload(args: argparse.Namespace) -> None:
    api = HfApi()
    entries = collect_entries(args)
    summary: list[dict[str, object]] = []
    for local_path, remote_path in entries:
        if not local_path.is_file():
            print(f"skip missing {local_path}")
            continue
        size_bytes = local_path.stat().st_size
        summary.append({"local": str(local_path), "remote": remote_path, "size_bytes": size_bytes})
        if args.list_only:
            continue
        mime = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=f"Upload {remote_path}",
        )
        print(f"uploaded {local_path} -> {remote_path} ({size_bytes} bytes, {mime})")
    print(json.dumps({"repo_id": args.repo_id, "repo_type": args.repo_type, "files": summary}, indent=2))


def download(args: argparse.Namespace) -> None:
    out_dir = Path(args.download_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = collect_entries(args)
    for _, remote_path in entries:
        if args.list_only:
            print(remote_path)
            continue
        local_target = out_dir / remote_path
        local_target.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            filename=remote_path,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
        print(f"downloaded {remote_path} -> {local_target}")


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "upload":
        upload(args)
    else:
        download(args)


if __name__ == "__main__":
    main()
