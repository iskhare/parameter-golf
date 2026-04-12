# Parameter Golf Handoff

## What We Found Locally On The Mac

All runs below used the MLX smoke path with:
- `ITERATIONS=20`
- `TRAIN_BATCH_TOKENS=8192`
- `VAL_BATCH_SIZE=524288`
- `MLX_MAX_MICROBATCH_TOKENS=2048`
- `SEED=1337`

Completed results:

| Profile | Key settings | `final_int8_zlib_roundtrip_exact val_bpb` | Notes |
|---|---|---:|---|
| `sp1024_baseline` | stock MLX baseline | `3.36399171` | weak local baseline |
| `sp4096_base` | just switch to `sp4096` | `3.05901520` | large immediate gain |
| `sp4096_mlp4_qk5` | `MLP_MULT=4`, `QK_GAIN_INIT=5.0` | `2.98311853` | clear gain over plain `sp4096` |
| `sp4096_mlp4_qk525_wd` | `MLP_MULT=4`, `QK_GAIN_INIT=5.25`, `WEIGHT_DECAY_EMBED=0.085`, `WEIGHT_DECAY_MATRIX=0.085`, `WEIGHT_DECAY_SCALAR=0.02` | `2.94634720` | **best local result so far** |
| `sp4096_recur` | above + `RECUR_LAYERS=4,5`, `RECUR_REPEAT_COUNT=1`, `RECUR_START_FRAC=0.50`, matrix WD `0.090` | `2.95276939` | slightly worse than no-recurrence on this smoke test |

Current local conclusion:
- The strongest direction is **not** `sp1024`.
- The best local branch is:
  - `sp4096`
  - `MLP_MULT=4`
  - `QK_GAIN_INIT=5.25`
  - split weight decay near `embed=0.085`, `matrix=0.085`, `scalar=0.02`
- The first recurrence port did **not** beat the non-recurrent version on this short Mac smoke test.

## Code Changes Already Made

In [train_gpt_mlx.py](/Users/ishan/Desktop/parameter-golf/train_gpt_mlx.py):
- added:
  - `WEIGHT_DECAY_EMBED`
  - `WEIGHT_DECAY_MATRIX`
  - `WEIGHT_DECAY_SCALAR`
  - `RECUR_LAYERS`
  - `RECUR_REPEAT_COUNT`
  - `RECUR_START_FRAC`
  - `PARALLEL_START_LAYER`
  - `SKIP_FINAL_PREQUANT_VAL`
- added train-side recurrence and parallel residual support
- added safe MLX version logging for this machine

In [run_mlx_experiments.py](/Users/ishan/Desktop/parameter-golf/run_mlx_experiments.py):
- reproducible local smoke runner
- JSONL result logging to [results.jsonl](/Users/ishan/Desktop/parameter-golf/logs/experiments/results.jsonl)

## What To Do On 8xH100 Now

Priority order:

1. Start from the **leaderboard-relevant family**, not `sp1024`.
   - Use `sp4096` first if you want the lowest implementation risk.
   - Promote to `sp8192` immediately after one clean sanity run if throughput/artifact size look fine.

2. First H100 run should test the current best local recipe:
   - `VOCAB_SIZE=4096`
   - `MLP_MULT=4`
   - `QK_GAIN_INIT=5.25`
   - weight decay split approximately:
     - embed `0.085`
     - matrix `0.085`
     - scalar `0.02`

3. Do **not** make recurrence the first H100 bet.
   - On the Mac smoke test, the recurrence port was slightly worse than the no-recurrence branch.
   - Recurrence is still worth testing later on H100 because longer training horizons may flip the ranking.

4. After the first clean H100 run, branch in this order:
   - `sp4096_mlp4_qk525_wd`
   - same config but `sp8192`
   - same config + recurrence / parallel residuals
   - only then try more aggressive stack fusion

5. Most likely best H100 promotion path:
   - take the current local best branch
   - merge it into the closest CUDA leaderboard lineage rather than relying on MLX parity
   - best target lineage is the modern `sp4096` / `sp8192` record family, not the baseline scripts

## Suggested H100 Experiment Order

1. `sp4096` + `MLP_MULT=4` + `QK_GAIN_INIT=5.25` + split WD
2. same but `sp8192`
3. same but recurrence on layers `4,5`
4. same but recurrence + parallel residuals
5. if all are clean, consider porting the winner into the full top-stack CUDA path with GPTQ / SDClip / other record-only pieces

## Important Practical Notes

- `sp4096` and `sp8192` datasets/tokenizers are already downloaded locally for smoke work.
- The Mac smoke runner disables the redundant final pre-quant validation pass, so H100-side timing should be evaluated separately from these local wallclocks.
- Full local result log is in [results.jsonl](/Users/ishan/Desktop/parameter-golf/logs/experiments/results.jsonl).
