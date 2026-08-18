# llm-inference-optimization

Phase 1 of an LLM inference optimization roadmap. One small model (Qwen2.5-1.5B,
fp16) run two ways on a single GPU, measured by one shared harness, then pushed
until the KV cache OOMs. Everything is measured on real rented hardware and the
raw CSVs are committed next to the code that produced them.

Four findings came out of it, and none of them were the expected ones:

**1. The card with the most memory bandwidth had the slowest baseline decode.**

| card | bandwidth | HF decode | HF MBU | vLLM decode | vLLM MBU | vLLM / HF |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 3060 12GB | 360 GB/s | 20.6 tok/s | 17.7% | 79.9 tok/s | 68.5% | 3.88x |
| RTX 4060 Ti 8GB | 288 GB/s | **70.1 tok/s** | 75.2% | 83.2 tok/s | 89.2% | 1.19x |
| RTX 3070 Ti 8GB | **608 GB/s** | 30.6 tok/s | 15.5% | **152.0 tok/s** | 77.2% | 4.97x |
| RTX 3070 8GB | 448 GB/s | 59.8 tok/s | 41.2% | 115.8 tok/s | 79.9% | 1.94x |

Ranked by bandwidth: 3070 Ti, 3070, 3060, 4060 Ti. Ranked by HuggingFace decode:
4060 Ti, 3070, 3070 Ti, 3060. Swap in vLLM and the ranking returns to bandwidth order.
Decode is memory-bound, but an eager Python decode loop on a slow host hits a
dispatch wall long before the bandwidth wall, and buying bandwidth buys nothing
until the engine can keep the GPU fed. vLLM's batch-1 win is not a fixed
multiplier: it equals the host overhead that was there to remove, which is why
the same swap pays 1.19x on a fast host and 4.97x on a slow one. This table is
confounded, see Known Gaps; the controlled version of the claim is finding 3.

**2. The KV-cache formula is exactly right; peak memory is not the cache.** Live
cache bytes track the analytical 28,672 B/token at 1.00x across ten checkpoints
out to 20k tokens, correct dtype, GQA collapsed, no padding. What the OOM curves
measured was peak, which carries a per-step reallocation transient. Switching
from the legacy tuple cache to a `DynamicCache` object drops peak growth from
92,317 to 63,595 B/token, exactly one full copy of the cache, which is 59,595
versus 86,509 tokens of context on an 8GB card. See `docs/kv-cache-growth.md`.

**3. Removing per-token CPU work on fixed hardware gave 1.687x.** One card, one
process, `StaticCache` in both arms, only the dispatch mechanism changed:
`torch.compile` with CUDA graphs took decode from 54.33 to 91.65 tok/s while
prefill stayed flat (65.36 to 66.97 ms). A profiler independently measured 1,282
kernels per token averaging 8.429 us, with the GPU idle 7.60 ms of every 18.41 ms
eager token and 0.10 ms of every compiled one. See `docs/baseline-results-rtx3070.md`.

**4. At high batch, decode never becomes compute-bound.** It peaks at 19% of
tensor-core peak while holding 79 to 90% of memory bandwidth throughout. What
ends the batching free lunch is the KV cache growing into the same bandwidth
budget the weights were using, reaching 43% of all traffic at batch 128.

Full writeups are in `docs/`, and `blog/draft.md` is the narrative version.

## Hardware

All measurements are on rented single-GPU Vast.ai boxes. **All three boxes have
since been destroyed**, so the committed CSVs are the durable artifact. Anyone
can rent equivalent hardware; nothing here depends on a specific machine.

| card | VRAM | bandwidth | host CPU | what was measured |
| --- | --- | --- | --- | --- |
| RTX 3060 | 12GB | 360 GB/s | not recorded | HF, vLLM, OOM sweep |
| RTX 4060 Ti | 8GB | 288 GB/s | i7-11700 | everything (canonical card) |
| RTX 3070 Ti | 8GB | 608 GB/s | Xeon E5-2680 v4 | HF, vLLM |
| RTX 3070 | 8GB | 448 GB/s spec, **402.6 measured** | Ryzen 7 3700X | HF, vLLM, dispatch proof, KV probe, profiler |

The 4060 Ti is the canonical card: all three walls (compute, memory, KV
capacity) were measured on the same silicon, and its baseline reproduces to 0.1%
across runs three days apart. The 3060's HF baseline varies 29% between its two
runs and should be treated as soft.

The harness did not record the host CPU at the time these ran, which in
hindsight was the single most important variable. It does now.

## What is here

```
scripts/
  bench_common.py       shared harness: timing, VRAM, CPU/GPU logging, KV math
  baseline_hf.py        HF transformers baseline (the control)
  bench_vllm.py         vLLM, same model and workload
  oom_sweep.py          push context length until CUDA OOMs
  prefill_sweep.py      prefill efficiency vs prompt length (the compute wall)
  bench_vllm_batch.py   vLLM throughput vs batch size (the KV wall)
  kv_growth_probe.py    decompose GPU memory during decode (live vs peak)
  dispatch_proof.py     eager vs torch.compile CUDA graphs on fixed hardware
  profile_decode.py     count kernels per token and measure GPU idle
  plot_oom_curve.py     redraw the OOM figure from a committed CSV
  plot_batch_sweep.py   redraw the batching figure from a committed CSV
  plot_kv_growth.py     redraw the KV decomposition figure
  plot_decode_budget.py redraw the per-token time budget figure
docs/                   per-card results and their interpretation
results/rtx3060/        raw CSVs and plots, one directory per card
results/rtx4060ti/
results/rtx3070ti/
results/rtx3070/
blog/draft.md           the writeup, grown alongside the measurements
test/probe_weights_vram.py   diagnostic: break resident VRAM down by dtype
```

The plotting scripts are deliberately free of torch, so every figure can be
regenerated on a laptop from the committed CSVs long after the box is gone:

```bash
python scripts/plot_oom_curve.py \
    --csv results/rtx4060ti/oom_sweep.csv \
    --plot results/rtx4060ti/oom_curve.png
```

## Measurement rules

Every script follows these. Most wrong inference numbers come from breaking one.

- Prefill and decode are timed separately and never blended.
- Decode tok/s is generated tokens divided by decode wall time, excluding the
  prompt and excluding prefill.
- `torch.cuda.synchronize()` before every timer read, since kernels launch
  asynchronously.
- A warmup generation runs first and is discarded.
- VRAM tracks allocated and reserved separately, with peaks reset per run.
- fp16 only, never bf16, for comparability across all cards including the later
  V100 and T4 runs.
- Identical model, prompt, `max_new_tokens`, greedy decoding, and batch size
  across engines. Only the engine changes.

## Reproducing

Two environments, on purpose. vLLM pins its own torch, and installing it into
the baseline environment would replace the torch the baseline was measured on.

**Environment A, the HF baseline:**

```bash
export HF_HOME=/workspace/hf-cache
pip install -r requirements.txt
python scripts/baseline_hf.py \
    --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --new-tokens 256 \
    --csv results/<yourcard>/baseline_hf.csv
```

**Environment B, vLLM:**

```bash
python -m venv vllm-env && source vllm-env/bin/activate
pip install uv && uv pip install vllm==0.24.0 --torch-backend=auto
python scripts/bench_vllm.py \
    --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --new-tokens 256 \
    --gpu-mem-util 0.9 --csv results/<yourcard>/baseline_vllm.csv
```

Result paths are required rather than defaulted, so every row lands in a
card-labeled directory. Mixing cards in one file is the easiest way to ruin this
dataset.

## Working dependency versions

These are the combinations that actually ran, read off the committed CSVs rather
than aspirational pins.

| | 4060 Ti | 3070 Ti | 3060 |
| --- | --- | --- | --- |
| HF env torch | 2.12.0+cu126 | 2.12.0+cu130 | 2.12.0+cu130 |
| vLLM env torch | 2.11.0+cu129 | 2.11.0+cu130 | 2.11.0+cu130 |
| vLLM | 0.24.0 | 0.24.0 | 0.24.0 |

transformers is pinned to 4.46.3. Earlier versions register per-layer rotary
buffers sized to `max_position_embeddings`, which wastes 1.79 GB of VRAM on
Qwen2.5 and inflates the apparent weight footprint. See `requirements.txt`.

All cards are compute capability 8.0 or newer, so vLLM's optimized attention
backends and FlashAttention are available with no fallback.

## Known gaps

Stated plainly rather than buried:

- **The vLLM advantage is not cleanly decomposed.** vLLM reaches 8.63 ms/token,
  below the 10.806 ms of GPU time profiled for the eager path, so it is running
  cheaper kernels and not only launching fewer. vLLM was never profiled, so the
  split between fusion and launch removal is not measured.
- **The host-CPU claim is supported but not proven across cards.** The
  eager-versus-CUDA-graph run on the RTX 3070 shows per-token CPU work cost 7.60
  ms on that machine, with the GPU held fixed. It does not establish that the
  cross-card throughput inversion was caused by the CPU: those rows still vary
  CPU, GPU, and torch build together, one of them is a single run, and three of
  the four CPUs were never recorded in the result file.
- **MBU denominators are inconsistent.** Only the RTX 3070 has a measured
  bandwidth figure (402.6 GB/s). The other three cards divide by vendor spec,
  which understates their MBU by roughly the 10% gap measured here.
- **Coverage is uneven across cards.** Only the 4060 Ti has the prefill and
  batching sweeps. The two boxes that could have filled the gaps no longer exist.

See `CLAUDE.md` for persistent project context and `docs/phase1.md` for the
original task breakdown.
