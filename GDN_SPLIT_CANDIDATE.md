# GDN Hybrid Split-Eval Candidate

Local frontier reproduction and improvement on the GDN-Hybrid + SWA + warmdown1000 family.

## Setup

- Base lineage: `2026-04-12_GDN_Hybrid_DeltaRule_Warmdown1000_CompressedCode_1.0167`
- Tokenizer / data: `sp1024`
- Architecture: `D_GDN_Hybrid`
- Training: `590s`, `WARMDOWN_ITERS=1000`, `EMA_DECAY=0.997`, `QK_GAIN_INIT=5.0`
- Workflow change: split training from post-train scoring
  - `train_gpt.py` used only to train and save raw EMA weights
  - `gdn_eval_only.py` used to run distributed EMA eval, GPTQ, quantized eval, and XSA eval

## Legal 3-seed candidate

| Seed | EMA BPB | Quantized BPB | XSA BPB | Artifact bytes |
|------|--------:|--------------:|--------:|---------------:|
| 42 | 1.006565 | 1.015019 | 1.020022 | 15,918,956 |
| 1337 | 1.007132 | 1.015806 | 1.020281 | 15,987,204 |
| 314 | 1.007333 | 1.016296 | 1.020578 | 15,922,957 |
| **Mean** | **1.007010** | **1.015707** | **1.020294** | **15,943,039** |

Quantized 3-seed mean:

`(1.015019 + 1.015806 + 1.016296) / 3 = 1.015707`

This is below the currently strongest open GDN record claim of `1.01671233`.

## Notes

- Seed `2024` reproduced near the published BPB (`1.018220`) but produced an oversized artifact (`16,035,799` bytes), so it is excluded from the legal 3-seed set.
- The split eval path also improved seed `42` relative to the published `1.016200` and preserved legality.
- Best follow-on single-seed tweak on this line was `QK_GAIN_INIT=5.25` with the unchanged frontier graph.
  - Raw EMA improved to `1.006351` on seed `42` (vs `1.006565` for the legal baseline seed `42`).
  - Default GPTQ calibration (`64` sequences) produced `1.015121` BPB but an oversized artifact (`16,000,507` bytes).
  - The best legal post-train setting found was `GPTQ_NUM_SEQS=56`, which produced a legal `15,999,204` byte artifact and `1.015104` quantized BPB.
  - That is slightly worse than the current legal seed `42` baseline (`1.015019`), so it was not promoted into the 3-seed set.
- Dead ends tested after the split workflow:
  - `EMA_DECAY=0.9975` worsened raw EMA (`1.008946` on seed `42`).
  - `SWA_WINDOW=1024` worsened BPB and overflowed the size budget.
  - `gdn5_swa_gdn5` plus `MLP_MULT=3.25` was decisively worse (`1.021197` EMA BPB on seed `42`).
- Candidate code added during this search:
  - `run_gdn_sweep.py`
  - `gdn_eval_only.py`
  - `run_gdn_split_sweep.py`
