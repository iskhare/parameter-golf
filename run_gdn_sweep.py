#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "ema_bpb": re.compile(r"EMA BPB \(no XSA\): (?P<value>\S+)"),
    "quantized_bpb": re.compile(r"Quantized BPB \(no XSA\): (?P<value>\S+)"),
    "xsa_bpb": re.compile(r"Quantized BPB \(XSA-all\): (?P<value>\S+)"),
    "artifact_bytes": re.compile(r"Artifact: (?P<value>[\d,]+) bytes"),
    "train_steps": re.compile(r"Training complete in \S+s \((?P<value>\d+) steps\)"),
}


BASE_ENV: dict[str, str] = {
    "ARCH_MODE": "D",
    "DATA_PATH": "/workspace/parameter-golf/data/datasets/fineweb10B_sp1024",
    "TOKENIZER_PATH": "/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
    "TRAIN_LOG_EVERY": "100",
    "VAL_LOSS_EVERY": "1000000",
    "SAVE_EVERY": "0",
    "ITERATIONS": "9999",
    "MAX_WALLCLOCK_SECONDS": "590",
    "WARMDOWN_ITERS": "1000",
    "EMA_DECAY": "0.997",
    "GPTQ_ENABLED": "1",
    "OMP_NUM_THREADS": "1",
}


PROFILES: dict[str, dict[str, str]] = {
    "gdn_exact": {},
    "qk_5p10": {"QK_GAIN_INIT": "5.10"},
    "qk_5p25": {"QK_GAIN_INIT": "5.25"},
    "qk_4p90": {"QK_GAIN_INIT": "4.90"},
    "warmdown_900": {"WARMDOWN_ITERS": "900"},
    "warmdown_1100": {"WARMDOWN_ITERS": "1100"},
    "ema_0p9965": {"EMA_DECAY": "0.9965"},
    "ema_0p9975": {"EMA_DECAY": "0.9975"},
    "batch_1024k": {"TRAIN_BATCH_TOKENS": "1048576"},
    "batch_896k": {"TRAIN_BATCH_TOKENS": "917504"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeatable 8xH100 sweeps on the frontier GDN stack.")
    parser.add_argument("--profiles", nargs="+", default=["gdn_exact"], choices=sorted(PROFILES))
    parser.add_argument("--train-script", default="/workspace/parameter-golf/tmp/gdn_frontier/train_gpt.py")
    parser.add_argument("--workdir", default="/workspace/parameter-golf/tmp/gdn_frontier")
    parser.add_argument("--results-jsonl", default="/workspace/parameter-golf/logs/gdn_sweeps/results.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_metrics(text: str) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        value = match.group("value").replace(",", "")
        out[key] = int(value) if key in {"artifact_bytes", "train_steps"} else float(value)
    return out


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_profile(args: argparse.Namespace, profile: str) -> dict[str, object]:
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(PROFILES[profile])
    run_id = f"{profile}_seed{args.seed}"
    ckpt_dir = f"/workspace/parameter-golf/tmp/gdn_frontier/checkpoints/{run_id}"
    env.update(
        {
            "RUN_ID": run_id,
            "SEED": str(args.seed),
            "CKPT_DIR": ckpt_dir,
        }
    )
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        args.train_script,
    ]
    row: dict[str, object] = {
        "profile": profile,
        "run_id": run_id,
        "seed": args.seed,
        "overrides": PROFILES[profile],
        "cmd": cmd,
    }
    if args.dry_run:
        row["env"] = {k: env[k] for k in sorted(set(BASE_ENV) | set(PROFILES[profile]) | {"RUN_ID", "SEED", "CKPT_DIR"})}
        return row

    started = time.time()
    proc = subprocess.run(cmd, cwd=args.workdir, env=env, capture_output=True, text=True)
    wallclock_s = round(time.time() - started, 3)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    row["returncode"] = proc.returncode
    row["wallclock_s"] = wallclock_s
    row["metrics"] = parse_metrics(proc.stdout)
    return row


def main() -> None:
    args = build_parser().parse_args()
    results_path = Path(args.results_jsonl)
    for profile in args.profiles:
        row = run_profile(args, profile)
        append_jsonl(results_path, row)
        if args.dry_run:
            print(json.dumps(row, indent=2, sort_keys=True))
            continue
        metrics = row.get("metrics", {})
        summary = f"profile:{profile} seed:{args.seed} returncode:{row['returncode']} wallclock_s:{row['wallclock_s']}"
        if isinstance(metrics, dict):
            if "quantized_bpb" in metrics:
                summary += f" quantized_bpb:{metrics['quantized_bpb']:.6f}"
            if "xsa_bpb" in metrics:
                summary += f" xsa_bpb:{metrics['xsa_bpb']:.6f}"
            if "artifact_bytes" in metrics:
                summary += f" artifact_bytes:{metrics['artifact_bytes']}"
        print(summary)


if __name__ == "__main__":
    main()
