# Non-Record: GDN Hybrid + Split Eval + Legal 3-Seed Packaging

**Author:** Ishan Khare ([@iskhare](https://github.com/iskhare))  
**Base lineage:** `2026-04-12_GDN_Hybrid_DeltaRule_Warmdown1000_CompressedCode_1.0167`  
**Track:** `track_non_record_16mb`

## Summary

This folder packages a legal 3-seed reproduction and small improvement of the recent GDN-Hybrid + sliding-window-attention + compressed-code line. The main engineering change was to split training from post-train scoring:

- `train_gpt.py` trains for the 590s budget and saves raw EMA weights
- `gdn_eval_only.py` reloads the EMA checkpoint and performs EMA eval, GPTQ, quantized eval, and optional XSA eval

This made it easier to iterate on quantization and artifact legality without retraining.

## Best Legal 3-Seed Result

| Seed | EMA BPB | Quantized BPB | XSA BPB | Artifact bytes |
|------|--------:|--------------:|--------:|---------------:|
| 42   | 1.006565 | 1.015019 | 1.020022 | 15,918,956 |
| 1337 | 1.007132 | 1.015806 | 1.020281 | 15,987,204 |
| 314  | 1.007333 | 1.016296 | 1.020578 | 15,922,957 |
| **Mean** | **1.007010** | **1.015707** | **1.020294** | **15,943,039** |

This is numerically below the open GDN PR claim of `1.01671233`, but it does **not** clear the repo’s record threshold, so it is packaged here as a non-record contribution.

## Additional Findings

- `QK_GAIN_INIT=5.25` improved raw EMA on seed `42` to `1.006351`, but the default GPTQ artifact was slightly oversized at `16,000,507` bytes.
- The best legal post-train retry for that branch used `GPTQ_NUM_SEQS=56`, giving `15,999,204` bytes and `1.015104` quantized BPB.
- `EMA_DECAY=0.9975`, `SWA_WINDOW=1024`, and a `gdn5_swa_gdn5 + MLP_MULT=3.25` variant were all worse than the exact frontier graph.

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `architectures.py`
- `configs.py`
- `gdn_eval_only.py`
- `train_seed42.log`, `train_seed1337.log`, `train_seed314.log`
- `eval_seed42.log`, `eval_seed1337.log`, `eval_seed314.log`
