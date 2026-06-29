# HF baseline (RTX 3070 Ti): measured numbers and what they mean

> **Hardware: NVIDIA RTX 3070 Ti 8GB (Ampere GA104, sm_86).** Every number in
> this file was measured on the current Vast.ai RTX 3070 Ti box. The RTX 3060
> numbers live in `baseline-hf-results.md` and the RTX 4060 Ti numbers under
> `results/rtx4060ti/`. Do not relabel across cards. The cross-card comparison
> below is the point of keeping all three.

Task 1 on the 3070 Ti. This is the control the 3070 Ti vLLM run is measured
against. The raw row is `results/rtx3070ti/baseline_hf.csv`; this file is the
reading of it. The headline here is not a number, it is a surprise: HF decode on
this box is **slower** than on the 4060 Ti despite the 3070 Ti having ~2x the
memory bandwidth, and the reason is the whole lesson of the phase.

## Setup

- GPU: RTX 3070 Ti 8GB, Ampere GA104, sm_86. ~21.7 fp32 TFLOPS, ~608 GB/s spec
  memory bandwidth (GDDR6X, 256-bit, 19 Gbps; the vast.ai listing shows 523.8,
  the 608 figure is the spec).
- Host: Xeon E5-2680 v4 (Broadwell, 2016), weak single-thread. This matters: see
  the decode section. The 4060 Ti box had a much faster i7-11700.
- Model: Qwen/Qwen2.5-1.5B, fp16, 28 layers, 12 query heads / 2 KV heads (GQA),
  head_dim 128. Same checkpoint and config as every other card.
- Stack: torch 2.12.0+cu130 (the box's preinstalled torch), transformers, eager
  mode. No CUDA graphs, no fusion.
- Workload: 512-token prompt, 256 new tokens, batch 1, greedy. Manual prefill +
  decode loop so the two phases are timed apart.

## Measured

| metric | RTX 3070 Ti HF | RTX 4060 Ti HF | RTX 3060 HF |
| --- | --- | --- | --- |
| prefill | 47.5 ms (10,774 tok/s) | 44.6 ms (11,488 tok/s) | ~137 ms (3,740 tok/s) |
| decode | **30.6 tok/s** | **70.0 tok/s** | ~18-23 tok/s |
| decode MBU | **~15.5%** | ~75% | ~15-20% |
| weights VRAM | 2945 MiB | 2945 MiB | 2945 MiB |
| peak allocated | 3133 MiB | 3133 MiB | 3133 MiB |
| peak reserved | 3360 MiB | 3360 MiB | 3360 MiB |

Weights, peak allocated, and peak reserved are byte-identical across all three
cards (shape-determined, not card-determined), which is a good harness
sanity-check: the same model loaded the same way everywhere.

## Prefill is compute-bound and CPU-insensitive

Forward-pass FLOPs are about `2 x params x tokens`:

```
2 x 1.544e9 x 512 = 1.58e12 FLOP
measured 47.5 ms -> 1.58e12 / 0.0475 = 33.2 TFLOPS effective
```

That 33.2 TFLOPS is ~38% of the card's fp16 tensor-core peak (~87 TFLOPS, FP32
accumulate, dense). Use the tensor peak, not the 21.7 fp32 number: at 21.7 the
floor would be 72.8 ms, slower than the 47.5 ms we measured, which is itself
proof the GEMMs ran on the tensor cores and not the fp32 CUDA cores. The gap from
the ~18 ms tensor floor (`1.58e12 / 87e12`) to 47.5 ms is GEMM-shape efficiency:
at M=512 the prefill matmul is only moderately sized, so cuBLAS leaves the tensor
cores half-empty (partial tiles, low occupancy). Larger prompts pack the tiles
and push effective TFLOPS toward the roof; `scripts/prefill_sweep.py` measures
that curve (effective TFLOPS vs prompt length), the compute-bound counterpart to
the decode-MBU and OOM sweeps.

The 3070 Ti prefill (47.5 ms) and the 4060 Ti prefill (44.6 ms) are within ~6%
of each other, and both are ~3x faster than the 3060 (137 ms). That tracks raw
compute: the 3070 Ti and 4060 Ti have similar tensor throughput and both far
outrun the 3060. Prefill barely moved between the two 8GB cards because it is one
big GEMM-heavy parallel pass: the GPU stays saturated and the CPU has time to
queue the next launch, so the slow Broadwell host does not bite. This is the
clean counterpart to the decode story below: the compute-bound phase ignores both
memory bandwidth and CPU speed.

## Decode is NOT memory-bound here, it is launch-bound

Decode is memory-bound *by nature* (arithmetic intensity ~1-2 FLOP/byte), so the
best case is to be limited by bandwidth. The floor is the time to stream the
weights once:

```
weights read per token = 2945 MiB = 3.09 GB
floor at 608 GB/s      = 3.09 / 608 = 5.08 ms/token  (~197 tok/s, 100% MBU)
measured 30.6 tok/s    = 32.7 ms/token -> 94 GB/s achieved -> 15.5% MBU
```

15.5% MBU means the 608 GB/s sits almost entirely idle. Decompose the 32.7 ms:
~5 ms is the real weight read, the other **~27.6 ms is overhead**. That overhead
is the CPU. One decode step in eager HF is ~500 separate CUDA kernels (28 layers
x ~15-20 kernels each: norms, projections, RoPE, attention, SiLU, residuals,
plus the LM head). At batch 1 each kernel is a vector-times-matrix that the GPU
finishes in a few microseconds, but the CPU needs ~5-15 us just to push each
launch through Python dispatch and the driver. So the GPU drains kernels faster
than the slow Xeon can issue them and sits idle waiting. Decode speed is capped
by **CPU launch rate, not by the card's bandwidth.**

This is why the 3070 Ti is slower at HF decode than the 4060 Ti despite 2x the
bandwidth: the GPU was never the bottleneck. The 4060 Ti looked faster (75% MBU,
70 tok/s) not because its GPU is better (it has *less* bandwidth) but because its
i7-11700 issues launches fast enough to nearly keep the pipe full. Same workload,
different host, different wall.

## One formula explains every decode number

```
decode_tok/s = MBU x memory_bandwidth / weight_bytes
               (engine + CPU)   (the card)
```

| run | MBU | x bandwidth | / 3.09 GB | = tok/s |
| --- | --- | --- | --- | --- |
| 3060 HF (slow host) | ~17% | 360 GB/s | | ~20 |
| 4060 Ti HF (fast host) | 75% | 288 GB/s | | 70 |
| 3070 Ti HF (slow host) | 15.5% | 608 GB/s | | 30.6 |

MBU is set by how much host overhead the engine carries; bandwidth is set by the
card. A fast card behind a slow host (this box under HF) wastes its bandwidth.
The fix is to stop launching ~500 kernels per token, which is exactly what vLLM
does (see `baseline-vllm-results-rtx3070ti.md`: same card, same Xeon, 77% MBU).

## A prediction that was wrong, and why

Going in, the expectation was ~2x the 4060 Ti decode (~140 tok/s) from 2x
bandwidth. It came in at 30.6, *below* the 4060 Ti. The prediction assumed HF
would reach the bandwidth wall, where the card sets the pace. On this box HF
reaches the *launch* wall first, where the slow CPU sets the pace, so doubling
bandwidth changed nothing. The ~140 figure was not wrong about the card: vLLM on
this same card hits 152 tok/s. It was wrong about which engine would expose the
bandwidth. That distinction is the result.

## Key learnings

- **Decode on this box is launch-bound, not bandwidth-bound.** At 15.5% MBU the
  608 GB/s bus is mostly idle; decode speed is capped by how fast the slow Xeon
  can issue ~500 kernels per token, not by the card.
- **A faster GPU does not help a launch-bound baseline.** The 3070 Ti is slower
  at HF decode than the 4060 Ti despite 2x the bandwidth, because the bottleneck
  was the host CPU, which is slower here, not the GPU.
- **One formula governs decode:** `tok/s = MBU x bandwidth / weight_bytes`. MBU
  is set by engine + CPU overhead, bandwidth by the card. Slow host -> low MBU
  -> wasted bandwidth.
- **Prefill is compute-bound and CPU-insensitive.** ~47 ms here vs ~45 ms on the
  4060 Ti: one big parallel pass keeps the GPU saturated, so the slow host does
  not bite.
- **"Memory-bound" is the workload's nature (its ceiling), not where you sit.**
  You can be stuck far below the bandwidth wall at the launch wall, which is
  exactly what HF does here.

## What carries into the next tasks

- Decode 15.5% MBU is the control. The 3070 Ti vLLM run over this is the headline
  comparison, and the gap is large precisely because the baseline is so far below
  the ceiling on this slow-host box.
- VRAM budget: 2945 MiB weights, ~5 GB free on the 8GB card for KV and
  activations. The OOM sweep can trust `8 GB - weights` with the same per-token
  KV cost (28 KB/token) as the other cards.
- "Decode is memory-bound" is the workload's nature, but the *measured* wall here
  is the CPU, not the bus. The phase's thesis is that an engine which keeps the
  GPU fed turns the launch wall back into the bandwidth wall.
