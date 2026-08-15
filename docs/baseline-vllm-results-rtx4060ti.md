# vLLM baseline (RTX 4060 Ti): measured numbers and what they mean

> **Hardware: NVIDIA RTX 4060 Ti 8GB (Ada AD106, sm_89).** Measured on the
> Vast.ai RTX 4060 Ti box, since destroyed. Raw row:
> `results/rtx4060ti/baseline_vllm.csv`. The 3070 Ti equivalent is
> `baseline-vllm-results-rtx3070ti.md`, the 3060 one is
> `baseline-vllm-results.md`.

Task 2 on the 4060 Ti, measured against the HF control in
`baseline-hf-results-rtx4060ti.md`. Same model, same 512-token prompt, same 256
new tokens, same batch 1, same greedy decoding. Only the engine changes.

The result is the most useful negative in the repo: **vLLM buys only 1.19x here.**
Every other card shows 3.9x to 5.0x. That is not vLLM underperforming, and it is
the cleanest evidence in the phase for what vLLM's batch-1 win actually is.

## Setup

- Engine: vLLM 0.24.0, offline `LLM` class in-process, `dtype="float16"`,
  `gpu_memory_utilization=0.9`, `max_model_len=784`, prefix caching disabled so
  the two-run prefill subtraction stays valid.
- Stack: torch 2.11.0+cu129 in the isolated `vllm-env` venv (Environment B), so
  vLLM's bundled torch cannot disturb the HF baseline's torch 2.12.0+cu126.
- Everything else identical to the HF run.

Note that the two engines run on different torch and CUDA builds. That is forced
by the two-environment design (vLLM pins its own torch), but it is an
uncontrolled variable in the comparison and worth stating rather than hiding.

## Measured

| metric | HF | vLLM | gap |
| --- | --- | --- | --- |
| prefill | 44.6 ms (11,488 tok/s) | 40.3 ms (12,708 tok/s) | 1.11x |
| decode | 70.1 tok/s | **83.2 tok/s** | **1.19x** |
| decode MBU | 75.2% | **89.2%** | |
| ms per token | 14.27 | 12.02 | |
| host overhead per token | 3.54 ms | **1.29 ms** | |
| VRAM | 3133 MiB grown | 7539 MiB reserved | not comparable |

## The 1.19x is the point

Decode has to move 3.088 GB of weights per token, and this card can move 288 GB/s,
so nothing can decode faster than 93.3 tok/s at batch 1:

```
ceiling (100% MBU) = 93.3 tok/s   (10.72 ms/token)
HF                 = 70.1 tok/s   (14.27 ms = 10.72 + 3.54 overhead)
vLLM               = 83.2 tok/s   (12.02 ms = 10.72 + 1.29 overhead)
```

vLLM did not make the GPU faster. It cut host overhead from 3.54 ms to 1.29 ms
per token, by capturing the decode step into a CUDA graph (one launch replays
every kernel in order) and fusing kernels so there are fewer to launch. The
remaining 1.29 ms is the irreducible per-token cost of a graph-replayed engine.
That figure is consistent across cards: 1.50 ms on the 3070 Ti, 1.29 ms here.

The win is small **because the baseline was already good**. HF on this fast host
was at 75.2% MBU, leaving only 24.8% of the bandwidth unclaimed, so even a
perfect engine could not have gained more than 1.33x. vLLM captured most of what
was available.

Set that against the same swap on the slow-host 3070 Ti box:

| | 4060 Ti (i7-11700) | 3070 Ti (Xeon E5-2680 v4) |
| --- | --- | --- |
| HF overhead per token | 3.54 ms | 27.64 ms |
| vLLM overhead per token | 1.29 ms | 1.50 ms |
| HF MBU | 75.2% | 15.5% |
| vLLM MBU | 89.2% | 77.2% |
| **vLLM / HF** | **1.19x** | **4.97x** |

vLLM lands both boxes at roughly the same small per-token overhead, 1.3 to 1.5
ms. What differs by 4x between the boxes is not where vLLM ends up, it is where
HF started. **vLLM's batch-1 speedup is not a property of vLLM. It is a
measurement of how much host overhead the baseline had.**

This is the most controlled comparison in the phase. Every cross-card claim here
varies CPU and GPU and torch build at once. This one changes only the engine, on
a fixed card, in a fixed box. Two such measurements, on two boxes, give the
host-overhead thesis a control that the raw cross-card table cannot.

## Prefill: 1.11x, and that is the expected answer

Both engines hand the same GEMMs to the same cuBLAS tensor-core kernels. There is
no matmul to beat. The 11% is trimmed launch overhead on a phase that was already
keeping the GPU busy, which is why prefill barely moves on every card measured
(1.10x to 1.13x). If prefill had jumped, something would be wrong with the
measurement.

## VRAM: a reserved pool, not comparable to HF

The 7539 MiB figure is NVML device-used memory at `gpu_memory_utilization=0.9`:
weights, plus the KV pool vLLM reserves in one slab at construction, plus
CUDA-graph buffers. It measures what vLLM grabbed up front, not what generation
needed. HF's 3133 MiB is what generation organically grew into. The two are
different measurements and putting them side by side without saying so would be
misleading. Compare engines on decode tok/s and prefill latency.

That reserved pool is also why vLLM never reproduces the HF OOM curve: memory is
flat from the first token, and running out of KV shows up as queued requests
rather than a crash. `batching-results.md` measures exactly that.

## Key learnings

- **vLLM's batch-1 win equals the host overhead removed, not a fixed multiplier.**
  1.19x on a fast host, 4.97x on a slow one, converging to the same ~1.3 ms
  per-token floor in both cases.
- **A near-ceiling baseline leaves nothing to win.** At 75.2% MBU the theoretical
  maximum gain was 1.33x; vLLM got 1.19x of it.
- **89.2% MBU is the highest number measured in this phase**, and it is on the
  card with the least bandwidth. Utilization and capability are different axes.
- **Prefill moves 1.11x, as on every other card.** Same cuBLAS kernels.
- **vLLM VRAM is a pre-grabbed pool.** Never compare it to HF organic growth.

## What carries into the next tasks

- The 1.19x / 4.97x contrast is the phase's controlled result. Lead the causal
  argument with it rather than with the cross-card table.
- vLLM's ~1.3 ms residual per-token overhead is the practical floor for batch-1
  decode. The 93.3 tok/s ceiling on this card is unreachable in practice.
- Batch 1 exercises none of PagedAttention or continuous batching. The reserved
  pool measured here is what those use, and `batching-results.md` is where it
  starts to matter.
