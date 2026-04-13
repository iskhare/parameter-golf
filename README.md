# Parameter Golf

This repository documents a focused attempt to push the OpenAI Parameter Golf challenge frontier: train the best language model that fits in a `16 MB` artifact and trains in under `10 minutes` on `8xH100`, evaluated by compression on FineWeb validation.

The main outcome here is not a new record. It is a clean reproduction and extension of the strongest recent GDN-Hybrid line, plus tooling that made it much easier to separate training from quantized scoring.

## Snapshot

- Best packaged result in this repo: **`1.015707` quantized bpb (3-seed mean)**  
  See [README.md](/workspace/parameter-golf/records/track_non_record_16mb/2026-04-13_GDN_Hybrid_SplitEval_Legal3Seed/README.md)
- Best legal seed-42 result on the packaged line: **`1.015019`**
- Best raw seed-42 follow-up checkpoint found during search: **`1.006351` EMA bpb** with `QK_GAIN_INIT=5.25`
- Key tooling added:
  - [gdn_eval_only.py](/workspace/parameter-golf/gdn_eval_only.py)
  - [run_gdn_sweep.py](/workspace/parameter-golf/run_gdn_sweep.py)
  - [run_gdn_split_sweep.py](/workspace/parameter-golf/run_gdn_split_sweep.py)

## Best Result In This Repo

The strongest packaged result is a legal 3-seed reproduction and improvement on the recent GDN-Hybrid + sliding-window-attention + warmdown1000 family.

| Seed | EMA BPB | Quantized BPB | XSA BPB | Artifact bytes |
|---|---:|---:|---:|---:|
| `42` | `1.006565` | `1.015019` | `1.020022` | `15,918,956` |
| `1337` | `1.007132` | `1.015806` | `1.020281` | `15,987,204` |
| `314` | `1.007333` | `1.016296` | `1.020578` | `15,922,957` |
| **Mean** | **`1.007010`** | **`1.015707`** | **`1.020294`** | **`15,943,039`** |

This result is preserved as a non-record package in [README.md](/workspace/parameter-golf/records/track_non_record_16mb/2026-04-13_GDN_Hybrid_SplitEval_Legal3Seed/README.md).

## What Actually Mattered

The central engineering improvement was to split training from scoring:

1. train for the fixed wallclock budget and save raw EMA weights
2. reload the checkpoint separately for EMA eval, GPTQ, compressed artifact creation, and optional XSA

That split made quantization and legality iteration much faster. It also made the real bottleneck obvious: the hard part was not training a better checkpoint, it was preserving that gain through GPTQ and under the `16,000,000` byte limit.

Important search outcomes:

- `QK_GAIN_INIT=5.25` improved the raw seed-42 checkpoint to `1.006351` EMA bpb
- default GPTQ on that branch produced `1.015121` bpb, but the artifact was illegal at `16,000,507` bytes
- the best legal post-train retry on that branch used `GPTQ_NUM_SEQS=56`, yielding `15,999,204` bytes and `1.015104` bpb
- `EMA_DECAY=0.9975`, `SWA_WINDOW=1024`, and a `gdn5_swa_gdn5 + MLP_MULT=3.25` variant all lost to the exact frontier graph

In practice, **quantization and artifact legality were the limiting factors**, not raw training throughput or missing CUDA kernels.

## Repo Structure

- [train_gpt.py](/workspace/parameter-golf/train_gpt.py): baseline PyTorch/CUDA training entry point
- [train_gpt_mlx.py](/workspace/parameter-golf/train_gpt_mlx.py): Apple MLX local experimentation path
- [BLOG.md](/workspace/parameter-golf/BLOG.md): narrative writeup of the search from MacBook MLX runs to near-frontier GDN work
- [GDN_SPLIT_CANDIDATE.md](/workspace/parameter-golf/GDN_SPLIT_CANDIDATE.md): condensed summary of the split-eval GDN search
- [records/track_10min_16mb](/workspace/parameter-golf/records/track_10min_16mb): official-style record submissions and references
- [records/track_non_record_16mb](/workspace/parameter-golf/records/track_non_record_16mb): non-record runs, ablations, and negative results
- [data/README.md](/workspace/parameter-golf/data/README.md): dataset and tokenizer cache details
- [AGENTS.md](/workspace/parameter-golf/AGENTS.md): contributor-oriented repo guide

## Packaged Contribution

The packaged non-record folder includes:

- exact source snapshot (`train_gpt.py`, `architectures.py`, `configs.py`, `gdn_eval_only.py`)
- train logs for seeds `42`, `1337`, and `314`
- eval logs for seeds `42`, `1337`, and `314`
- a `submission.json` matching the repo’s records layout

See [README.md](/workspace/parameter-golf/records/track_non_record_16mb/2026-04-13_GDN_Hybrid_SplitEval_Legal3Seed/README.md).

## Minimal Reproduction Pointers

If you want to reproduce or extend this work:

- use [AGENTS.md](/workspace/parameter-golf/AGENTS.md) for the practical repo workflow
- use [data/README.md](/workspace/parameter-golf/data/README.md) for dataset/tokenizer setup
- start from the packaged GDN folder instead of the repo root baseline if your goal is frontier chasing

For local iteration, the MLX path remains useful for ranking ideas cheaply before spending H100 time.

## Bottom Line

This repo got close enough to the current GDN frontier to expose the real constraint: **a better raw checkpoint is not enough if the gain disappears after GPTQ and compression**.

That is the main lesson here, and the reason the most valuable artifacts are the split-eval tooling and the fully packaged non-record result rather than a new leaderboard entry.
