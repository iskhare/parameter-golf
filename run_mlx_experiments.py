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


FINAL_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>\S+) val_bpb:(?P<val_bpb>\S+)"
)


PRESETS: dict[str, dict[str, str]] = {
    "sp1024_baseline": {
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
    },
    "sp4096_base": {
        "DATA_PATH": "./data/datasets/fineweb10B_sp4096",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_4096_bpe.model",
        "VOCAB_SIZE": "4096",
    },
    "sp4096_mlp4_qk5": {
        "DATA_PATH": "./data/datasets/fineweb10B_sp4096",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_4096_bpe.model",
        "VOCAB_SIZE": "4096",
        "MLP_MULT": "4",
        "QK_GAIN_INIT": "5.0",
    },
    "sp4096_mlp4_qk525_wd": {
        "DATA_PATH": "./data/datasets/fineweb10B_sp4096",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_4096_bpe.model",
        "VOCAB_SIZE": "4096",
        "MLP_MULT": "4",
        "QK_GAIN_INIT": "5.25",
        "WEIGHT_DECAY_EMBED": "0.085",
        "WEIGHT_DECAY_MATRIX": "0.085",
        "WEIGHT_DECAY_SCALAR": "0.02",
    },
    "sp4096_recur": {
        "DATA_PATH": "./data/datasets/fineweb10B_sp4096",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_4096_bpe.model",
        "VOCAB_SIZE": "4096",
        "MLP_MULT": "4",
        "QK_GAIN_INIT": "5.25",
        "WEIGHT_DECAY_EMBED": "0.085",
        "WEIGHT_DECAY_MATRIX": "0.090",
        "WEIGHT_DECAY_SCALAR": "0.02",
        "RECUR_LAYERS": "4,5",
        "RECUR_REPEAT_COUNT": "1",
        "RECUR_START_FRAC": "0.50",
    },
    "sp4096_recur_parallel": {
        "DATA_PATH": "./data/datasets/fineweb10B_sp4096",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_4096_bpe.model",
        "VOCAB_SIZE": "4096",
        "MLP_MULT": "4",
        "QK_GAIN_INIT": "5.25",
        "WEIGHT_DECAY_EMBED": "0.085",
        "WEIGHT_DECAY_MATRIX": "0.090",
        "WEIGHT_DECAY_SCALAR": "0.02",
        "RECUR_LAYERS": "4,5",
        "RECUR_REPEAT_COUNT": "1",
        "RECUR_START_FRAC": "0.50",
        "PARALLEL_START_LAYER": "7",
    },
    "sp8192_promote": {
        "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
        "VOCAB_SIZE": "8192",
        "MLP_MULT": "4",
        "QK_GAIN_INIT": "5.25",
        "WEIGHT_DECAY_EMBED": "0.085",
        "WEIGHT_DECAY_MATRIX": "0.090",
        "WEIGHT_DECAY_SCALAR": "0.02",
        "RECUR_LAYERS": "4,5",
        "RECUR_REPEAT_COUNT": "1",
        "RECUR_START_FRAC": "0.50",
        "PARALLEL_START_LAYER": "7",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible MLX smoke experiments.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["sp1024_baseline", "sp4096_base", "sp4096_mlp4_qk5"],
        choices=sorted(PRESETS),
        help="Ordered preset list to execute.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--train-batch-tokens", type=int, default=8192)
    parser.add_argument("--val-batch-size", type=int, default=524288)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--mlx-max-microbatch-tokens", type=int, default=4096)
    parser.add_argument("--train-log-every", type=int, default=25)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", default="logs/experiments")
    parser.add_argument("--results-jsonl", default="logs/experiments/results.jsonl")
    return parser


def run_profile(args: argparse.Namespace, profile: str) -> dict[str, object]:
    env = os.environ.copy()
    env.update(PRESETS[profile])
    env.update(
        {
            "RUN_ID": f"{profile}_seed{args.seed}_iter{args.iterations}",
            "ITERATIONS": str(args.iterations),
            "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
            "VAL_BATCH_SIZE": str(args.val_batch_size),
            "GRAD_ACCUM_STEPS": str(args.grad_accum_steps),
            "MLX_MAX_MICROBATCH_TOKENS": str(args.mlx_max_microbatch_tokens),
            "TRAIN_LOG_EVERY": str(args.train_log_every),
            "WARMUP_STEPS": str(args.warmup_steps),
            "VAL_LOSS_EVERY": "0",
            "OUT_DIR": args.out_dir,
            "SEED": str(args.seed),
        }
    )
    cmd = [args.python, "train_gpt_mlx.py"]
    started_at = time.time()
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=env, capture_output=True, text=True)
    wallclock_s = time.time() - started_at
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    match = FINAL_RE.search(proc.stdout)
    result: dict[str, object] = {
        "profile": profile,
        "returncode": proc.returncode,
        "wallclock_s": round(wallclock_s, 3),
        "run_id": env["RUN_ID"],
        "seed": args.seed,
        "iterations": args.iterations,
        "train_batch_tokens": args.train_batch_tokens,
        "val_batch_size": args.val_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "mlx_max_microbatch_tokens": args.mlx_max_microbatch_tokens,
        "env": PRESETS[profile],
    }
    if match:
        result["val_loss"] = float(match.group("val_loss"))
        result["val_bpb"] = float(match.group("val_bpb"))
    else:
        result["val_loss"] = None
        result["val_bpb"] = None
    if proc.returncode != 0:
        result["error"] = "train_gpt_mlx.py failed"
    return result


def append_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    results_path = Path(args.results_jsonl)
    for profile in args.profiles:
        row = run_profile(args, profile)
        append_result(results_path, row)
        summary = f"profile:{profile} returncode:{row['returncode']} wallclock_s:{row['wallclock_s']}"
        if row["val_bpb"] is not None:
            summary += f" val_loss:{row['val_loss']:.8f} val_bpb:{row['val_bpb']:.8f}"
        print(summary)


if __name__ == "__main__":
    main()
