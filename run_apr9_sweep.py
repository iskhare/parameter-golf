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


METRIC_RE = re.compile(
    r"(?P<label>quantized|quantized_sliding_window|quantized_ttt) "
    r"val_loss:(?P<val_loss>\S+) val_bpb:(?P<val_bpb>\S+) eval_time:(?P<eval_ms>\d+)ms"
)


BASE_ENV: dict[str, str] = {
    "VOCAB_SIZE": "8192",
    "NUM_LAYERS": "11",
    "MODEL_DIM": "512",
    "EMBEDDING_DIM": "512",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "MLP_MULT": "4",
    "TRAIN_BATCH_TOKENS": "786432",
    "TRAIN_SEQ_LEN": "2048",
    "EVAL_SEQ_LEN": "2048",
    "VAL_BATCH_TOKENS": "524288",
    "EVAL_STRIDE": "64",
    "QK_GAIN_INIT": "5.25",
    "NUM_LOOPS": "2",
    "LOOP_START": "3",
    "LOOP_END": "5",
    "ENABLE_LOOPING_AT": "0.35",
    "PARALLEL_RESIDUAL_START": "7",
    "MATRIX_LR": "0.022",
    "SCALAR_LR": "0.02",
    "TIED_EMBED_LR": "0.03",
    "MUON_WD": "0.095",
    "EMBED_WD": "0.085",
    "ADAM_WD": "0.02",
    "EMA_DECAY": "0.9965",
    "MUON_MOMENTUM": "0.99",
    "MUON_MOMENTUM_WARMUP_START": "0.92",
    "MUON_MOMENTUM_WARMUP_STEPS": "1500",
    "GRAD_CLIP_NORM": "0.3",
    "MATRIX_BITS": "6",
    "EMBED_BITS": "8",
    "MATRIX_CLIP_SIGMAS": "12.85",
    "EMBED_CLIP_SIGMAS": "20.0",
    "GPTQ_CALIBRATION_BATCHES": "64",
    "GPTQ_RESERVE_SECONDS": "12",
    "COMPRESSOR": "brotli",
    "RUN_PREQUANT_EVAL": "0",
    "RUN_QUANT_EVAL": "0",
    "RUN_SLIDING_EVAL": "1",
    "RUN_TTT_EVAL": "0",
    "RUN_ETLB_EVAL": "0",
    "SLIDING_WINDOW_ENABLED": "1",
    "TTT_ENABLED": "0",
    "TRAIN_LOG_EVERY": "100",
    "VAL_LOSS_EVERY": "0",
}


PROFILES: dict[str, dict[str, str]] = {
    "apr9_baseline": {},
    "qk_5p35": {"QK_GAIN_INIT": "5.35"},
    "qk_5p40": {"QK_GAIN_INIT": "5.40"},
    "qk_5p50": {"QK_GAIN_INIT": "5.50"},
    "muw_0p092": {"MUON_WD": "0.092"},
    "muw_0p098": {"MUON_WD": "0.098"},
    "ema_0p9970": {"EMA_DECAY": "0.9970"},
    "qk_5p35_ema_0p9970": {"QK_GAIN_INIT": "5.35", "EMA_DECAY": "0.9970"},
    "qk_5p35_clip_13p20": {"QK_GAIN_INIT": "5.35", "MATRIX_CLIP_SIGMAS": "13.20"},
    "late_loop": {"ENABLE_LOOPING_AT": "0.45"},
    "clip_13p20": {"MATRIX_CLIP_SIGMAS": "13.20"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch repeatable 8xH100 sweeps on the April 9 stack.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["apr9_baseline", "qk_5p35", "muw_0p092"],
        choices=sorted(PROFILES),
        help="Ordered profile list to execute.",
    )
    parser.add_argument("--train-script", default="research_apr9_train_gpt.py")
    parser.add_argument("--results-jsonl", default="logs/apr9_sweeps/results.jsonl")
    parser.add_argument("--iterations", type=int, default=2200)
    parser.add_argument("--max-wallclock-seconds", type=float, default=210.0)
    parser.add_argument("--gptq-calibration-batches", type=int, default=16)
    parser.add_argument("--val-tokens-limit", type=int, default=4_194_304)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--ttt", action="store_true", help="Enable final TTT eval for selected profiles.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_metrics(stdout: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for match in METRIC_RE.finditer(stdout):
        metrics[match.group("label")] = {
            "val_loss": float(match.group("val_loss")),
            "val_bpb": float(match.group("val_bpb")),
            "eval_ms": float(match.group("eval_ms")),
        }
    return metrics


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_profile(args: argparse.Namespace, profile: str) -> dict[str, object]:
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(PROFILES[profile])
    env.update(
        {
            "RUN_ID": f"{profile}_seed{args.seed}_iter{args.iterations}",
            "ITERATIONS": str(args.iterations),
            "MAX_WALLCLOCK_SECONDS": str(args.max_wallclock_seconds),
            "GPTQ_CALIBRATION_BATCHES": str(args.gptq_calibration_batches),
            "VAL_TOKENS_LIMIT": str(args.val_tokens_limit),
            "SEED": str(args.seed),
            "TTT_ENABLED": "1" if args.ttt else env["TTT_ENABLED"],
            "RUN_TTT_EVAL": "1" if args.ttt else env["RUN_TTT_EVAL"],
        }
    )
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        args.train_script,
    ]
    result: dict[str, object] = {
        "profile": profile,
        "run_id": env["RUN_ID"],
        "seed": args.seed,
        "iterations": args.iterations,
        "max_wallclock_seconds": args.max_wallclock_seconds,
            "gptq_calibration_batches": args.gptq_calibration_batches,
            "val_tokens_limit": args.val_tokens_limit,
            "overrides": PROFILES[profile],
        }
    if args.dry_run:
        result["cmd"] = cmd
        result["env"] = {k: env[k] for k in sorted(BASE_ENV | PROFILES[profile])}
        return result
    started = time.time()
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=env, capture_output=True, text=True)
    wallclock_s = round(time.time() - started, 3)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    metrics = parse_metrics(proc.stdout)
    result.update(
        {
            "returncode": proc.returncode,
            "wallclock_s": wallclock_s,
            "metrics": metrics,
        }
    )
    for label, values in metrics.items():
        result[f"{label}_val_bpb"] = values["val_bpb"]
    return result


def main() -> None:
    args = build_parser().parse_args()
    results_path = Path(args.results_jsonl)
    for profile in args.profiles:
        row = run_profile(args, profile)
        append_jsonl(results_path, row)
        if args.dry_run:
            print(json.dumps(row, indent=2, sort_keys=True))
            continue
        summary = f"profile:{profile} returncode:{row['returncode']} wallclock_s:{row['wallclock_s']}"
        if "quantized_sliding_window_val_bpb" in row:
            summary += f" sliding_bpb:{row['quantized_sliding_window_val_bpb']:.8f}"
        if "quantized_ttt_val_bpb" in row:
            summary += f" ttt_bpb:{row['quantized_ttt_val_bpb']:.8f}"
        print(summary)


if __name__ == "__main__":
    main()
