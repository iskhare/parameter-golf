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
    "POST_EVAL_RANK0_ONLY": "1",
    "EXIT_AFTER_EMA_SAVE": "1",
    "OMP_NUM_THREADS": "1",
}


PROFILES: dict[str, dict[str, str]] = {
    "baseline": {},
    "qk_5p25": {"QK_GAIN_INIT": "5.25"},
    "wd_0p06": {"MUON_WD": "0.06", "ADAM_WD": "0.06"},
    "qk_5p25_wd_0p06": {"QK_GAIN_INIT": "5.25", "MUON_WD": "0.06", "ADAM_WD": "0.06"},
    "layout_swa1_mlp3p25": {"LAYER_LAYOUT": "gdn5_swa_gdn5", "MLP_MULT": "3.25"},
    "layout_swa1_mlp3p25_qk_5p25": {
        "LAYER_LAYOUT": "gdn5_swa_gdn5",
        "MLP_MULT": "3.25",
        "QK_GAIN_INIT": "5.25",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run split train/eval sweeps on the GDN frontier stack.")
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-script", default="/workspace/parameter-golf/tmp/gdn_frontier/train_gpt.py")
    parser.add_argument("--eval-script", default="/workspace/parameter-golf/gdn_eval_only.py")
    parser.add_argument("--workdir", default="/workspace/parameter-golf")
    parser.add_argument("--results-jsonl", default="/workspace/parameter-golf/logs/gdn_split_sweeps/results.jsonl")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--gptq-num-seqs", type=int, default=64)
    parser.add_argument("--gptq-batch-size", type=int, default=8)
    parser.add_argument("--gptq-temperature", type=float, default=0.8)
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


def run_cmd(cmd: list[str], env: dict[str, str], cwd: str) -> tuple[int, float, str, str]:
    started = time.time()
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    wallclock_s = round(time.time() - started, 3)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    return proc.returncode, wallclock_s, proc.stdout, proc.stderr


def main() -> None:
    args = build_parser().parse_args()
    results_path = Path(args.results_jsonl)

    for profile in args.profiles:
        overrides = PROFILES[profile]
        run_id = f"{profile}_seed{args.seed}"
        train_ckpt_dir = f"/workspace/parameter-golf/tmp/gdn_frontier/checkpoints/{run_id}"
        eval_ckpt_dir = f"/workspace/parameter-golf/tmp/gdn_frontier/eval_artifacts/{run_id}"
        model_path = f"{train_ckpt_dir}/final_model_D_GDN_Hybrid_seed{args.seed}.pt"

        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(overrides)
        env.update(
            {
                "RUN_ID": run_id,
                "SEED": str(args.seed),
                "CKPT_DIR": train_ckpt_dir,
            }
        )

        train_cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            args.train_script,
        ]
        eval_cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            args.eval_script,
            "--model-path",
            model_path,
            "--run-id",
            run_id,
            "--ckpt-dir",
            eval_ckpt_dir,
            "--seed",
            str(args.seed),
            "--gptq-num-seqs",
            str(args.gptq_num_seqs),
            "--gptq-batch-size",
            str(args.gptq_batch_size),
            "--gptq-temperature",
            str(args.gptq_temperature),
        ]
        if "QK_GAIN_INIT" in overrides:
            eval_cmd.extend(["--qk-gain-init", overrides["QK_GAIN_INIT"]])
        if "SWA_WINDOW" in overrides:
            eval_cmd.extend(["--swa-window", overrides["SWA_WINDOW"]])
        if "BIGRAM_VOCAB_SIZE" in overrides:
            eval_cmd.extend(["--bigram-vocab-size", overrides["BIGRAM_VOCAB_SIZE"]])
        if "BIGRAM_DIM" in overrides:
            eval_cmd.extend(["--bigram-dim", overrides["BIGRAM_DIM"]])
        if "TRIGRAM" in overrides:
            eval_cmd.extend(["--trigram", overrides["TRIGRAM"]])
        if "MLP_MULT" in overrides:
            eval_cmd.extend(["--mlp-mult", overrides["MLP_MULT"]])
        if "LAYER_LAYOUT" in overrides:
            eval_cmd.extend(["--layer-layout", overrides["LAYER_LAYOUT"]])

        row: dict[str, object] = {
            "profile": profile,
            "run_id": run_id,
            "seed": args.seed,
            "overrides": overrides,
            "train_cmd": train_cmd,
            "eval_cmd": eval_cmd,
        }

        train_rc, train_wallclock_s, train_stdout, train_stderr = run_cmd(train_cmd, env, args.workdir)
        row["train_returncode"] = train_rc
        row["train_wallclock_s"] = train_wallclock_s
        row["train_metrics"] = parse_metrics(train_stdout)
        if train_rc != 0:
            row["status"] = "train_failed"
            row["stderr_tail"] = train_stderr[-2000:]
            append_jsonl(results_path, row)
            print(f"profile:{profile} seed:{args.seed} train_failed rc:{train_rc}")
            continue

        eval_env = os.environ.copy()
        eval_env["OMP_NUM_THREADS"] = "1"
        eval_rc, eval_wallclock_s, eval_stdout, eval_stderr = run_cmd(eval_cmd, eval_env, args.workdir)
        row["eval_returncode"] = eval_rc
        row["eval_wallclock_s"] = eval_wallclock_s
        row["eval_metrics"] = parse_metrics(eval_stdout)
        row["status"] = "ok" if eval_rc == 0 else "eval_failed"
        if eval_rc != 0:
            row["stderr_tail"] = eval_stderr[-2000:]
        append_jsonl(results_path, row)

        metrics = row["eval_metrics"] if isinstance(row["eval_metrics"], dict) else {}
        summary = (
            f"profile:{profile} seed:{args.seed} status:{row['status']} "
            f"train_s:{train_wallclock_s:.1f} eval_s:{eval_wallclock_s:.1f}"
        )
        if "ema_bpb" in metrics:
            summary += f" ema_bpb:{metrics['ema_bpb']:.6f}"
        if "quantized_bpb" in metrics:
            summary += f" quantized_bpb:{metrics['quantized_bpb']:.6f}"
        if "artifact_bytes" in metrics:
            summary += f" artifact_bytes:{metrics['artifact_bytes']}"
        print(summary)


if __name__ == "__main__":
    main()
