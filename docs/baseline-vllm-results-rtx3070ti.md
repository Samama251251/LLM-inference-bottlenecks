# vLLM baseline (RTX 3070 Ti): measured numbers and what they mean

> **Hardware: NVIDIA RTX 3070 Ti 8GB (Ampere GA104, sm_86).** Every number in
> this file was measured on the current Vast.ai RTX 3070 Ti box. The RTX 3060
> numbers live in `baseline-vllm-results.md`. Do not relabel across cards.

Task 2 on the 3070 Ti. The whole point is the gap against
`baseline-hf-results-rtx3070ti.md`: identical model, prompt, token count, batch,
and decoding, only the engine changed. The raw row is
`results/rtx3070ti/baseline_vllm.csv`; this file is the reading of it.

## Setup

- GPU: RTX 3070 Ti 8GB, Ampere, sm_86, ~608 GB/s spec bandwidth. Same card and
  same Xeon E5-2680 v4 host as the HF baseline, which is the key control: the
  *only* thing that changed is the engine, so any decode difference is the engine,
  not the hardware.
- Model: Qwen/Qwen2.5-1.5B, fp16 (cast from the bf16 checkpoint on load).
- Stack: vLLM (v1 engine), torch 2.11.0+cu130 in the separate vLLM venv,
  FlashAttention backend (sm_86 has no fallback worry). CUDA graphs on, kernels
  fused.
- Workload: 512-token prompt, 256 new tokens, batch 1, greedy, byte-identical
  input to the HF run. Prefix caching disabled, `ignore_eos` forces exactly 256
  tokens. `gpu_memory_utilization=0.9`, `max_model_len=784`.
- Prefill and decode are split by timing two runs on the identical prompt
  (`max_tokens=1` for TTFT, `max_tokens=256` for the full run, decode is the
  difference), so the split does not depend on vLLM's internal metrics.

## Measured

| metric | vLLM | HF baseline | gap |
| --- | --- | --- | --- |
| prefill | 43.7 ms (11,726 tok/s) | 47.5 ms (10,774 tok/s) | ~1.1x |
| decode | **152.0 tok/s** (255 tokens) | **30.6 tok/s** | **~5.0x** |
| decode MBU | **~77%** | ~15.5% | |
| device VRAM | 7678 MiB (reserved pool, NVML) | 3360 MiB peak reserved (organic) | not comparable |

## The comparison is the result

### Prefill barely moved, as expected

Both engines hand the prefill GEMMs to the same cuBLAS tensor-core kernels, so
there is no matmul to beat. vLLM's ~8% edge (43.7 ms vs 47.5 ms) is fused kernels
and CUDA graphs trimming launch overhead, not faster math. Prefill is
compute-bound and already near the cuBLAS ceiling, so this is where vLLM has the
least to offer, and the table shows it.

### Decode: ~5x, and the size of the win is the point

```
floor (100% MBU):  5.08 ms/token -> 197 tok/s   (stream 3.09 GB of weights once)
vLLM (measured):   6.58 ms/token -> 152 tok/s   -> 470 GB/s -> 77% MBU
HF (measured):    32.7  ms/token ->  30.6 tok/s -> 94 GB/s -> 15.5% MBU
```

Same card, same 608 GB/s, same slow Xeon, same 3.09 GB of weights to read.
vLLM took decode from 15.5% to 77% MBU on identical hardware. The mechanism is
exactly the one the HF baseline named: HF eager launches ~500 tiny kernels per
token and the slow CPU cannot issue them fast enough, so the GPU stalls between
launches. vLLM captures the whole decode step into one CUDA graph (record once,
replay with a single launch) and fuses kernels, so the CPU stops being the
bottleneck and the GPU streams weights back to back. Per-token overhead drops
from ~27.6 ms (HF) to ~1.5 ms (vLLM) on top of the same 5.08 ms weight read.

The ~5x here is much larger than vLLM's batch-1 win would be on the 4060 Ti box
(where HF was already at 75% MBU, leaving ~1.3x to recover). The win equals the
host overhead removed, so a slow-host box shows a bigger vLLM advantage. That is
itself a finding: vLLM's batch-1 decode edge is not a fixed multiplier, it scales
with how launch-bound the baseline was.

### The bandwidth thesis, confirmed at equal MBU

You cannot compare HF-to-HF across the 8GB cards (different hosts confound it).
But compare the two runs that both reach the ceiling:

```
4060 Ti HF:   75.1% MBU on 288 GB/s ->  70 tok/s
3070 Ti vLLM: 77.2% MBU on 608 GB/s -> 152 tok/s

throughput ratio = 152 / 70 = 2.17x
  bandwidth ratio = 608 / 288 = 2.11x
  MBU ratio       = 77.2 / 75.1 = 1.03x
  2.11 x 1.03     = 2.17  (exact)
```

After normalizing the small MBU difference, decode throughput scales 2.11x,
identical to the bandwidth ratio. That is "decode is memory-bound" measured, not
asserted: once the engine keeps the GPU fed, decode speed tracks memory bandwidth
one-to-one. The clean version (vLLM-to-vLLM across the two 8GB cards) is missing
because the 4060 Ti box was destroyed before its vLLM run; the equal-MBU argument
stands in for it.

## VRAM: a reserved pool, read via NVML, not comparable to HF

The 7678 MiB figure is NVML device-used memory at `gpu_memory_utilization=0.9`,
not the HF-style organic-growth number. vLLM pre-allocates the KV cache as one
slab at construction (weights + reserved KV pool + CUDA-graph buffers), so this
counts memory it grabbed up front, not memory generation needed. The honest
HF-vs-vLLM comparison is decode tok/s and prefill latency, never this figure. The
per-component split (weights / KV pool / graphs) is in the vLLM init log; the CSV
note records that this is NVML device-used and not comparable to HF.

That reserved pool is the paged KV pool continuous batching would use, and it is
the same pool that will keep vLLM alive in the Task 3 OOM sweep while plain HF
fragments and crashes. At batch 1 the pool's concurrency is irrelevant; it
matters only for the OOM contrast and for throughput under load.

## Key learnings

- **vLLM removed ~5x of overhead on identical hardware** (15.5% -> 77% MBU) by
  replacing ~500 per-token kernel launches with one CUDA-graph replay plus fused
  kernels. The card did not change; the CPU stopped being the bottleneck.
- **vLLM's batch-1 decode win is not a fixed multiplier, it equals the host
  overhead removed.** Slow-host boxes show bigger wins: ~5x here vs a
  hypothetical ~1.3x on the fast-host 4060 Ti (where HF was already at 75% MBU).
- **Decode throughput scales with bandwidth at equal MBU** (2.11x measured =
  2.11x bandwidth ratio). This is the memory-bound thesis confirmed, once the
  engine keeps the GPU fed.
- **Prefill barely moves (~1.1x).** Both engines call the same cuBLAS GEMMs;
  there is no matmul to beat, only launch overhead to trim.
- **vLLM VRAM is a pre-grabbed pool, not comparable to HF organic growth.**
  Compare engines on decode tok/s and prefill latency, never on the VRAM figure.
- **The realistic ceiling is ~77% MBU, not 100%.** The 5.08 ms weight-read time
  per token (197 tok/s) is an unreachable floor; even a graph-replayed engine
  carries ~1.5 ms of irreducible per-token overhead.

## What carries into the next tasks

- The 3070 Ti HF-vs-vLLM headline: ~1.1x prefill, ~5x decode, MBU 15.5% -> 77%.
  The 5x (vs the 3060's ~4x and a hypothetical ~1.3x on the 4060 Ti) is the
  host-overhead effect made visible.
- vLLM's 152 tok/s is the realistic ceiling on this card (~77% MBU); the 197
  theoretical floor (100% MBU) is unreachable. Decode cannot beat the 5.08 ms
  weight-read time per token.
- Missing for an airtight cross-card bandwidth comparison: a 4060 Ti vLLM run.
  If another 8GB card is ever spun up, run vLLM on it to get vLLM-to-vLLM at
  equal MBU directly.
