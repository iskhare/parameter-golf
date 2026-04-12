# QK 5.35 Candidate

Candidate lineage: `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py`

Single change:
- `QK_GAIN_INIT=5.35` instead of `5.25`

Full 8xH100 SXM results collected on 2026-04-12:

| Seed | Sliding BPB | TTT BPB |
| --- | ---: | ---: |
| 1337 | 1.08088579 | 1.07958567 |
| 42 | 1.08173642 | 1.08039600 |
| 314 | 1.08278267 | 1.08136141 |
| Mean | 1.08180163 | 1.08044769 |

Population std for TTT BPB: `0.00072586`

Current leaderboard target from `README.md`:
- `1.0810` for `SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT`

This candidate is provisionally better on the 3-seed mean:
- `1.0810 - 1.08044769 = 0.00055231 BPB`

Relevant logs:
- `logs/qk535_full_seed1337.txt`
- `logs/qk535_full_seed42.txt`
- `logs/qk535_full_seed314.txt`

Supporting search code:
- `research_apr9_train_gpt.py`
- `run_apr9_sweep.py`
- `data/cached_challenge_fineweb.py`
