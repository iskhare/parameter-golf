# Hunting the Parameter Golf Record

For a brief stretch, I had a simple goal: take a MacBook, a few local MLX experiments, a rented 8xH100 box, and try to climb to the top of the Parameter Golf leaderboard.

I did not get the record.

But I got close enough to understand where the real bottlenecks were, and the path from "local smoke test" to "near-frontier legal submission" turned out to be much more interesting than I expected.

This is the writeup I wish I had before starting.

## The Setup

Parameter Golf is a nasty optimization problem disguised as a leaderboard. You are not just trying to train a good language model. You are trying to train one that:

- fits under a strict artifact size cap
- trains within a fixed wallclock budget
- survives quantization
- holds up across multiple seeds
- and, for a true record claim, beats the previous best by a meaningful margin

That last part matters. It is not enough to sneak a tiny numerical improvement past the current best run. You need a result that is statistically convincing and clears the competition threshold.

So the process quickly becomes less about "what architecture looks cool?" and more about "what survives the full submission path?"

## Starting on a MacBook

Before touching expensive GPUs, I used the local MLX path on my MacBook to get directional signal.

The first important result was that `sp1024` was not the most promising lane. On short smoke tests, `sp4096` immediately looked better. The local progression went roughly like this:

| Profile | Key changes | `val_bpb` |
|---|---|---:|
| `sp1024_baseline` | stock baseline | `3.36399171` |
| `sp4096_base` | just switch vocab | `3.05901520` |
| `sp4096_mlp4_qk5` | `MLP_MULT=4`, `QK_GAIN_INIT=5.0` | `2.98311853` |
| `sp4096_mlp4_qk525_wd` | `MLP_MULT=4`, `QK_GAIN_INIT=5.25`, tuned weight decay | `2.94634720` |

That was enough to establish a basic pattern:

1. Vocabulary choice mattered a lot.
2. QK gain mattered.
3. Weight decay splits mattered.
4. Tiny local runs were still useful, as long as I treated them as ranking signals rather than absolute truth.

I also tested a first recurrence port locally. It did not beat the non-recurrent branch. That ended up being a useful early warning: recurrence was interesting, but it was not obviously a free win.

## Moving to 8xH100

Once I moved to 8xH100, the search changed character.

At first I thought the problem would be architecture. Maybe a clever new block, maybe a different attention layout, maybe some custom kernel work. But after reading the stronger leaderboard PRs and reproducing the most relevant GDN-Hybrid line, it became clear that the real frontier was more constrained than that.

The best recent submissions were already using strong kernels, strong training recipes, and highly optimized quantization paths. Flash attention was already there. Muon was already there. The obvious systems wins had mostly been harvested.

So I switched focus from "invent a brand-new stack" to "take the strongest open lineage, reproduce it exactly, and then find changes that survive the entire legal submission pipeline."

That decision was correct.

## Reproducing the GDN Frontier

The family I focused on was the GDN-Hybrid + sliding-window-attention + warmdown1000 line.

I pulled that stack into a local frontier branch and reproduced the core setup:

- `D_GDN_Hybrid`
- `sp1024`
- `QK_GAIN_INIT=5.0`
- `WARMDOWN_ITERS=1000`
- `EMA_DECAY=0.997`
- `590s` training budget on `8x H100`
- GPTQ int6 + zstd artifact compression

Then I made one structural change that ended up being the most useful engineering improvement of the whole sprint: I split training from scoring.

Instead of doing everything in one giant end-to-end run, I changed the flow to:

1. train only
2. save raw EMA weights
3. run evaluation, GPTQ, compressed artifact creation, and optional XSA as a separate pass

That sounds mundane, but it changed the speed of iteration a lot. Once I had the raw EMA checkpoint, I could retry quantization settings, compression-sensitive knobs, and legality fixes without spending another 590 seconds on training.

This was the birth of the split-eval tooling now in the repo:

- `gdn_eval_only.py`
- `run_gdn_sweep.py`
- `run_gdn_split_sweep.py`

## The First Real Win

With the split workflow in place, I got a legal 3-seed candidate:

| Seed | EMA BPB | Quantized BPB | XSA BPB | Artifact bytes |
|---|---:|---:|---:|---:|
| `42` | `1.006565` | `1.015019` | `1.020022` | `15,918,956` |
| `1337` | `1.007132` | `1.015806` | `1.020281` | `15,987,204` |
| `314` | `1.007333` | `1.016296` | `1.020578` | `15,922,957` |
| **Mean** | **`1.007010`** | **`1.015707`** | **`1.020294`** | **`15,943,039`** |

Numerically, that was better than one of the strongest open GDN PRs at the time.

But not enough.

This was the painful part of Parameter Golf: you can improve a line, beat a visible number, and still not actually have a record-worthy submission. The competition threshold meant that a tiny edge was not enough. To claim the true top spot, I effectively needed something much closer to `1.0095` than `1.0157`.

That was the moment the problem became clear. I was not chasing tenths anymore. I was chasing a qualitatively stronger idea.

## Where the Search Failed

I tested several branches that looked plausible on paper and mostly failed for the same reason: they did not survive quantization and artifact packaging.

Some examples:

- `EMA_DECAY=0.9975` improved nothing meaningful and worsened raw EMA.
- `SWA_WINDOW=1024` hurt BPB and pushed the artifact over the size budget.
- A leaner `gdn5_swa_gdn5 + MLP_MULT=3.25` architecture looked elegant but was much worse in practice.
- Longer eval context was not a hidden cheat code either; `EVAL_SEQ_LEN=4096` made things worse.

The biggest lesson was that the bottleneck was not raw training loss.

The bottleneck was the gap between:

- the raw EMA checkpoint
- the quantized checkpoint
- and the compressed, legal artifact under `16 MB`

That gap is where good ideas went to die.

## The Most Frustrating Near-Miss

The single most annoying result of the whole sprint came from a very small change: `QK_GAIN_INIT=5.25` on the unchanged frontier graph.

This was not some exotic architecture leap. It was just a clean, controlled, exact-graph run with one tuned parameter.

And it actually worked.

On seed `42`, the raw EMA score improved from:

- baseline: `1.006565`
- `QK_GAIN_INIT=5.25`: `1.006351`

That is exactly the kind of move you want to see. Same graph. Same throughput. Better checkpoint.

Then quantization happened.

The default GPTQ path gave:

- quantized BPB: `1.015121`
- artifact size: `16,000,507` bytes

That artifact was illegal by `507` bytes.

Not 50 KB. Not 5 KB. Five hundred and seven bytes.

So I started doing minimal post-train legality surgery:

- change GPTQ calibration seed
- reduce calibration sequence count
- adjust calibration temperature

The best legal retry I found was:

- `GPTQ_NUM_SEQS=56`
- artifact size: `15,999,204`
- quantized BPB: `1.015104`

That was legal.

But it was still slightly worse than the existing legal seed-42 baseline of `1.015019`.

That one sequence of experiments explains the entire sprint better than anything else. I had a genuinely better raw model. I had an almost-legal artifact. I had a legal rescue. And in the final scored form, it still was not enough.

## What I Learned

Three things stand out.

### 1. The record was not hiding in custom kernels

I spent some time considering whether the answer was lower-level systems work. It was not.

The strong line already had the right kernel story. The real opportunity was ML-quantization co-design, not another round of hand-tuned CUDA.

### 2. Quantization is not a post-processing step

In this competition, quantization is part of the model design.

If a training tweak improves the raw checkpoint but makes the compressed artifact worse, it is not a real improvement. If a model gets better only before GPTQ, it does not count.

### 3. Good tooling matters even when you lose

I did not get the record, but I did end up with a cleaner search workflow:

- exact frontier reproduction
- split train/eval tooling
- easier artifact legality retries
- a packaged non-record result with logs and source snapshot

That work compounds. A lot of leaderboard chasing is really infrastructure work wearing a research hat.

## The Final State

I did not beat the best record-worthy submission.

What I did get was:

- a legal 3-seed GDN package at `1.015707`
- a stronger understanding of where the GDN frontier actually breaks
- and a very clear answer to the question "why didn’t this cross the line?"

It did not cross the line because the frontier was already too optimized for shallow tweaks to matter. The remaining gap was living inside quantization, compression, and legality, and I did not find the idea strong enough to survive all three at once.

That is disappointing.

It is also, I think, the honest shape of this kind of work. Most of the time, research is not a dramatic breakthrough. It is a sequence of increasingly specific failures, until the real constraint becomes visible.

For this sprint, that constraint was clear by the end:

the hard part was never training a better checkpoint.

The hard part was making a better checkpoint stay better after it became a legal submission.
