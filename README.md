# llm-inference-optimization

Phase 1 of an LLM inference optimization roadmap. One small model (Qwen2.5-1.5B,
fp16) run two ways on a single GPU, measured by one shared harness, then pushed
until the KV cache OOMs. Everything is measured on real rented hardware and the
raw CSVs are committed next to the code that produced them.

Three findings came out of it, and none of them were the expected ones:

**1. The card with the most memory bandwidth had the slowest baseline decode.**

| card | bandwidth | HF decode | HF MBU | vLLM decode | vLLM MBU | vLLM / HF |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 3060 12GB | 360 GB/s | 20.6 tok/s | 17.7% | 79.9 tok/s | 68.5% | 3.88x |
| RTX 4060 Ti 8GB | 288 GB/s | **70.1 tok/s** | 75.2% | 83.2 tok/s | 89.2% | 1.19x |
| RTX 3070 Ti 8GB | **608 GB/s** | 30.6 tok/s | 15.5% | **152.0 tok/s** | 77.2% | 4.97x |

Ranked by bandwidth: 3070 Ti, 3060, 4060 Ti. Ranked by HuggingFace decode:
4060 Ti, 3070 Ti, 3060. Swap in vLLM and the ranking returns to bandwidth order.
Decode is memory-bound, but an eager Python decode loop on a slow host hits a
dispatch wall long before the bandwidth wall, and buying bandwidth buys nothing
until the engine can keep the GPU fed. vLLM's batch-1 win is not a fixed
multiplier: it equals the host overhead that was there to remove, which is why
the same swap pays 1.19x on a fast host and 4.97x on a slow one.

**2. Measured KV-cache growth is 2.21x the analytical prediction.** Textbook math
says 28,672 bytes per token for this model. Both cards measured 63,4xx bytes per
token, agreeing to three significant figures, so it is systematic rather than
noise. The mechanism is not yet identified. This is the phase's main open
question and it is marked as such on the figure.

**3. At high batch, decode never becomes compute-bound.** It peaks at 19% of
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
  plot_oom_curve.py     redraw the OOM figure from a committed CSV
  plot_batch_sweep.py   redraw the batching figure from a committed CSV
docs/                   per-card results and their interpretation
results/rtx3060/        raw CSVs and plots, one directory per card
results/rtx4060ti/
results/rtx3070ti/
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

- **The 2.21x KV growth gap is unexplained.** It needs one instrumented run that
  breaks the allocation down per tensor.
- **The host-CPU claim is not fully controlled.** Every cross-card comparison
  varies CPU, GPU, and torch build together. The within-box engine swap (1.19x
  fast host vs 4.97x slow host) is the controlled evidence and is what the causal
  argument rests on. Settling it needs two boxes with the same GPU and different
  CPUs, or an eager-versus-CUDA-graph run on one card.
- **The ~500 kernels per decode token figure is an estimate**, derived from the
  layer count, not a profiled measurement.
- **Coverage is uneven across cards.** Only the 4060 Ti has the prefill and
  batching sweeps. The two boxes that could have filled the gaps no longer exist.

See `CLAUDE.md` for persistent project context and `docs/phase1.md` for the
original task breakdown.
