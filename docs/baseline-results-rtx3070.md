# RTX 3070: baselines, and the dispatch wall measured under control

> **Hardware: NVIDIA RTX 3070 8GB (Ampere GA104, sm_86, 46 SMs), AMD Ryzen 7
> 3700X host, 5.3 of 16 vCPU allocated.** Vast.ai instance 48034918, image
> `vastai/pytorch_cuda-12.8.1-auto`. Raw rows under `results/rtx3070/`. Do not
> relabel across cards.

This box was rented to answer two questions the earlier cards left open: whether
the host CPU really sets eager decode throughput, and why measured KV growth was
2.21x the analytical prediction. The KV answer is in `kv-cache-growth.md`. This
file covers the baselines and the dispatch experiment.

## Setup

- GPU: RTX 3070, 8192 MiB total, 7840 MiB visible to torch, sm_86, 46 SMs.
- Host: AMD Ryzen 7 3700X (Zen 2, 2019), 16 logical cores, 5.3 allocated. This
  is a shared allocation and it shows up in the run-to-run spread below.
- Model: Qwen/Qwen2.5-1.5B, fp16, 28 layers, 12 query heads, 2 KV heads,
  head_dim 128, weights 2945.29 MiB resident (3.088 GB).
- Environment A: torch 2.11.0+cu128, transformers 4.46.3, eager.
- Environment B: torch 2.11.0+cu130, vllm 0.24.0, separate venv.
- Workload: 512-token prompt, 255 decoded tokens, batch 1, greedy.

## Bandwidth was measured, not assumed

Every MBU number divides by bandwidth, so the denominator matters. The card is
specified at 448 GB/s. The Vast.ai dashboard reported 385.8 GB/s, and earlier in
the same session reported 179.6 GB/s for the same machine, which showed the
dashboard figure tracks host load.

A large contiguous fp16 copy, best of 20 iterations, measured **402.6 GB/s** of
combined read and write traffic, 90% of spec. Streaming 3.088 GB of weights at
that rate takes 7.671 ms, so batch-1 decode on this card cannot exceed 130.4
tok/s.

This is the only card in the repo with a measured bandwidth figure. The other
three divide by vendor spec, which understates their MBU by roughly the 10% gap
seen here. Cross-card MBU comparisons carry that inconsistency.

## Measured

| engine | decode | ms/token | MBU (402.6 GB/s) | host overhead |
| --- | --- | --- | --- | --- |
| HF eager | 59.75 tok/s | 16.74 | 45.8% | 9.07 ms |
| HF + `torch.compile` | 91.65 tok/s | 10.91 | 70.3% | 3.24 ms |
| vLLM 0.24.0 | 115.84 tok/s | 8.63 | 88.9% | 0.96 ms |

Medians. HF over five runs (55.07, 59.76, 58.03, 60.25, 59.75), vLLM over three
(115.96, 115.57, 115.84), compiled over three (92.87, 91.19, 91.65).

Prefill: HF 60.51 ms, vLLM 52.79 ms.

The HF runs span 9.4%, the first being slowest at 55.07 and the rest falling
between 58.03 and 60.25. vLLM on the same box span 0.34%. I ran five HF repeats
rather than the usual two because of the shared vCPU allocation, and the
difference in spread between the two engines is consistent with the noise
entering through host-side work, though I did not test that directly.

## The dispatch experiment

Every cross-card comparison in this repo varies CPU, GPU, and torch build
together, so none of them can show on its own that the host limits eager decode.
This one holds everything fixed except how work reaches the GPU: one card, one
box, one process, one model.

Both arms use a preallocated `StaticCache`. CUDA graphs require static shapes,
and a cache that grows by reallocation changes shape every step, so the compiled
arm needs it. Using it in the eager arm too keeps the only difference between
them the dispatch mechanism.

| arm | decode | ms/token | MBU | host overhead |
| --- | --- | --- | --- | --- |
| eager | 54.33 tok/s | 18.41 | 41.68% | 10.73 ms |
| compiled, CUDA graphs | 91.65 tok/s | 10.91 | 70.30% | 3.24 ms |

**1.687x with the hardware unchanged.** Prefill moved from 65.36 ms to 66.97 ms,
so the compiled path was slightly slower there, which is what a launch-overhead
explanation predicts: prefill is one large parallel pass that already keeps the
GPU busy, leaving no gap to close.

`StaticCache` is not free. Eager decode with it ran 54.33 tok/s against 59.75
with the normal cache, about 9% slower. So 1.687x is the speedup inside a
controlled comparison, and measured against the fastest eager configuration the
honest figure is 91.65 / 59.75 = **1.53x**. Both belong in any writeup.

## The profiler agrees from the other direction

`scripts/profile_decode.py` profiled twelve steady-state decode steps, counting
device-side kernels only.

| quantity | measured |
| --- | --- |
| CUDA kernels per decode token | **1,282** |
| GPU busy per token | 10.806 ms |
| average GPU time per kernel | 8.429 us |

Largest contributor is a `gemv2T_kernel_val` fp16 kernel, 57 launches per token
totalling 5.019 ms, followed by `cutlass_80_wmma_tensorop_f16` at 56 launches and
2.479 ms.

Set that against the wall-clock numbers:

| arm | wall | GPU busy | GPU idle |
| --- | --- | --- | --- |
| eager | 18.41 ms | 10.806 ms | 7.60 ms (41%) |
| compiled | 10.91 ms | 10.806 ms | 0.10 ms (1%) |

The compiled wall lands within 0.1 ms of the GPU's own busy time. Two methods
sharing no code agree that eager decode leaves the card idle 41% of each token
and graph replay closes it to roughly nothing.

This replaces the "on the order of 500 kernels per token" estimate that appeared
in earlier docs. That was a guess from the layer count. The measured value is
1,282, and the correction strengthens the argument rather than weakening it:
smaller average kernels make launch cost matter more.

## vLLM does something the compiled path does not

vLLM reached 115.84 tok/s, 8.63 ms per token. That is **below the 10.806 ms of
GPU time the profiler measured for the eager path**. Fewer launches cannot
explain a wall shorter than the GPU work itself, so vLLM is also running cheaper
kernels, presumably fused ones.

That splits the HF-to-vLLM gap into two effects rather than one:

- eager to compiled: dispatch removed, 7.60 ms of idle down to 0.10 ms.
- compiled to vLLM: 10.91 ms to 8.63 ms, with the GPU work itself getting
  cheaper.

Earlier docs in this repo attributed the whole gap to launch overhead. At least
part of it is kernel efficiency, and these measurements do not separate the two
cleanly, because vLLM was never profiled.

## Key learnings

- Removing per-token CPU work on fixed hardware gave 1.687x under control, 1.53x
  against the best eager configuration. The GPU did not change.
- Eager decode leaves the GPU idle 7.60 ms of every 18.41 ms token, 41%.
- One decode token costs 1,282 kernel launches averaging 8.429 us of GPU time.
- vLLM's 8.63 ms beats the eager path's 10.806 ms of GPU busy time, so fusion is
  doing work that launch removal alone cannot explain.
- Measured bandwidth was 402.6 GB/s against 448 spec and a dashboard figure that
  moved between 179.6 and 385.8 across one session. Measure the denominator.

## What this does not show

The 5.3-of-16 vCPU allocation means another tenant could have moved the eager
numbers, and the 9.4% spread across five runs is consistent with that.

Batch 1, one model, one prompt length, one card. Batch 1 is where launch overhead
matters most because GPU work per kernel is smallest, so none of this transfers
to throughput under load.

The dispatch result shows per-token CPU work cost 7.60 ms on this machine. It
does not establish what it costs elsewhere, and it does not prove the cross-card
throughput inversion was caused by the CPU, since those comparisons remain
confounded.

Environments A and B run different CUDA builds (cu128 and cu130), which is forced
by vLLM pinning its own torch but is an uncontrolled variable in the HF-vs-vLLM
row.
