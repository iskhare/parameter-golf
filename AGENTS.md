# Repository Guidelines

## Project Structure & Module Organization
Top-level training entry points live in `train_gpt.py` (PyTorch/CUDA) and `train_gpt_mlx.py` (Apple MLX). Utility scripts such as `run_mlx_experiments.py` and `sync_hf_artifacts.py` sit at the repo root. Dataset download, tokenizer, and export helpers live under `data/`; see `data/README.md` for cache and manifest details. Submission artifacts belong in `records/track_10min_16mb/` or `records/track_non_record_16mb/`, with one dated folder per run containing its own `README.md`, logs, and reproduction files.

## Build, Test, and Development Commands
Create an environment and install dependencies with `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
Download baseline data with `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.
Run a local CUDA baseline with `RUN_ID=dev torchrun --standalone --nproc_per_node=1 train_gpt.py`.
Run an MLX smoke test with `RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 python3 train_gpt_mlx.py`.
Batch MLX presets with `python3 run_mlx_experiments.py --profiles sp1024_baseline sp4096_base`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints where already used, and concise comments only around non-obvious logic. Keep environment-driven hyperparameters uppercase (`TRAIN_BATCH_TOKENS`, `VAL_LOSS_EVERY`) and Python identifiers `snake_case`. Record directories use `YYYY-MM-DD_<short_description>/`. The root training scripts are meant to stay readable for newcomers; keep changes focused and avoid large framework-style abstractions.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes with a small smoke run against cached data, then confirm final metrics/log output still includes `val_loss`, `val_bpb`, and compressed artifact reporting. For data tooling, run the relevant script on a minimal shard count and verify output paths under `data/datasets/` or `data/tokenizers/`.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects such as `Add HF artifact sync helper` and `Update README leaderboard for April records`. Keep commit titles concise and specific. PRs should explain the motivation, list exact commands used for validation, and link any related issue or leaderboard PR. For submissions, only add a new folder under `records/...` and include the required `README.md`, `submission.json`, train logs, and runnable training code.
