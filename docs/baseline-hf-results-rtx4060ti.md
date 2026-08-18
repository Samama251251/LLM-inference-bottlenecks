# HF baseline (RTX 4060 Ti): measured numbers and what they mean

> **Hardware: NVIDIA RTX 4060 Ti 8GB (Ada AD106, sm_89).** Every number in this
> file was measured on the Vast.ai RTX 4060 Ti box, which has since been
> destroyed. The raw rows survive in `results/rtx4060ti/`. The RTX 3060 numbers
> live in `baseline-hf-results.md` and the RTX 3070 Ti numbers in
> `baseline-hf-results-rtx3070ti.md`. Do not relabel across cards.

Task 1 on the 4060 Ti. Raw row: `results/rtx4060ti/baseline_hf.csv`.

This card matters out of proportion to its specs. It has the **least** memory
bandwidth of the three cards measured and the **fastest** HuggingFace decode, by
a wide margin. It is the only box where the eager baseline gets anywhere near the
memory-bandwidth wall, which makes it the control that gives the other two cards
their meaning: without it, the slow numbers elsewhere look like normal HF
overhead rather than a host bottleneck.

It is also the most completely measured card in the repo. All three walls
(compute, memory, KV capacity) were measured on this one piece of silicon, which
is why it is the canonical card for the writeup.

## Setup

- GPU: RTX 4060 Ti 8GB, Ada Lovelace AD106, sm_89. ~22.1 fp32 TFLOPS, ~88
  TFLOPS fp16 tensor (FP32 accumulate, dense), 288 GB/s memory bandwidth
  (GDDR6, 128-bit).
- Host: Intel i7-11700 (Rocket Lake, 2021). Strong single-thread. This is the
  variable that ends up mattering most, for reasons the decode section covers.
- Model: Qwen/Qwen2.5-1.5B, fp16, 28 layers, 12 query heads / 2 KV heads (GQA),
  head_dim 128. Same checkpoint and config as every other card.
- Stack: torch 2.12.0+cu126, transformers 4.46.3, eager mode. No CUDA graphs, no
  fusion, no compile.
- Workload: 512-token prompt, 256 new tokens, batch 1, greedy. Manual prefill and
  decode loop so the two phases are timed apart.

## Measured

| metric | RTX 4060 Ti HF | RTX 3070 Ti HF | RTX 3060 HF |
| --- | --- | --- | --- |
| memory bandwidth | 288 GB/s | 608 GB/s | 360 GB/s |
| prefill | 44.6 ms (11,488 tok/s) | 47.5 ms (10,774 tok/s) | ~137 ms (3,740 tok/s) |
| decode | **70.1 tok/s** | 30.6 tok/s | ~18-23 tok/s |
| decode MBU | **75.2%** | 15.5% | 17.7% |
| weights VRAM | 2945 MiB | 2945 MiB | 2945 MiB |
| peak allocated | 3133 MiB | 3133 MiB | 3133 MiB |
| peak reserved | 3360 MiB | 3360 MiB | 3360 MiB |

Two runs three days apart gave 70.04 and 70.13 tok/s, a spread of 0.1%. That is
the stability `phase1.md` asks for before any of this is trusted, and it is worth
contrasting with the 3060 control, whose two runs differ by 29% (18.0 and 23.2).
When reading cross-card claims, the 4060 Ti and 3070 Ti numbers are solid and the
3060 number is soft.

Weights, peak allocated, and peak reserved are byte-identical across all three
cards. Those are shape-determined, not card-determined, so agreement to the byte
is a harness sanity check: the same model really did load the same way everywhere.

## Decode: the only baseline that gets near the wall

Decode at batch 1 has to stream every weight once per token. In fp16 that is
2945 MiB, or 3.088 GB, so the card's bandwidth sets a hard ceiling:

```
weight read per token = 3.088 GB
ceiling  = 288 GB/s / 3.088 GB = 93.3 tok/s   (10.72 ms/token)
measured = 70.1 tok/s                          (14.27 ms/token)
MBU      = 70.1 / 93.3 = 75.2%
```

The eager HuggingFace loop leaves only **3.54 ms per token** of host overhead on
top of the unavoidable 10.72 ms weight read. That is the number to carry
forward. Every decode token re-enters Python, walks all 28 layers, and dispatches
1,282 kernels (measured on the RTX 3070 with torch.profiler, see
`baseline-results-rtx3070.md`), and on this box all of that fits into 3.5 ms.

The comparison that makes the point is the same code, same model, same
transformers version, on the 3070 Ti box:

| | 4060 Ti (i7-11700) | 3070 Ti (Xeon E5-2680 v4) |
| --- | --- | --- |
| weight read (unavoidable) | 10.72 ms | 5.08 ms |
| host overhead | **3.54 ms** | **27.64 ms** |
| total per token | 14.27 ms | 32.72 ms |
| decode | 70.1 tok/s | 30.6 tok/s |

The 3070 Ti's GPU does its half of the work in less than half the time, because
it has twice the bandwidth. It loses anyway, because its host takes 7.8x longer
to feed it. This is the whole finding of the phase, and this card is the half of
it that shows what the workload looks like when the host is not the problem.

One honest caveat on that 7.8x. An i7-11700 is roughly 2x an E5-2680 v4 in
single-thread, so raw clock does not explain a 7.8x overhead gap on its own.
"Host" here is shorthand for the whole dispatch path: CPU single-thread speed,
but also PCIe 3.0 x8 on the Xeon box, 14 of 28 vCPU allocated on a shared rented
machine with whatever contention that brings, and a different CUDA build (cu126
vs cu130). The direction is not in doubt and the mechanism is not in doubt, but
attributing the full 7.8x to the CPU alone would overstate what two confounded
boxes can prove. Closing that properly needs either two boxes with the same GPU
and different CPUs, or an eager-versus-CUDA-graph run on a single card.

## Prefill is compute-bound, and barely moves across cards

```
FLOPs  = 2 x 1.544e9 params x 512 tokens = 1.58e12
measured 44.6 ms -> 35.5 TFLOPS effective = 40% of the ~88 TFLOPS tensor peak
```

Use the tensor-core peak, not the 22.1 fp32 number: at 22.1 TFLOPS the floor
would be 71 ms, slower than the 44.6 ms measured, which is itself proof the GEMMs
ran on tensor cores.

Prefill is 44.6 ms here against 47.5 ms on the 3070 Ti, a 6% difference, while
decode differs by 2.3x. That asymmetry is the tell. Prefill is one long parallel
pass that keeps the GPU busy on its own, so a slow host has nothing to starve.
Decode is thousands of tiny sequential steps, and each one waits on Python.

The gap from the ~18 ms tensor floor to the measured 44.6 ms is GEMM-shape
efficiency, not host overhead: at M=512 the prefill matmul leaves tensor cores
partly idle. `scripts/prefill_sweep.py` walks that curve on this card and finds
efficiency climbing from 26% at 128 tokens to 46% at 8192 and then saturating.
See `results/rtx4060ti/prefill_curve.png`.

## Key learnings

- **This baseline is bandwidth-bound, not launch-bound, at 75.2% MBU.** It is the
  only one of the three that is. Everything a normal inference article says about
  decode being memory-bound is true on this box and misleading on the other two.
- **Host overhead is 3.54 ms per token here and 27.64 ms on the 3070 Ti box.**
  Same code, same model, same transformers version. The host, not the card,
  decides whether the bandwidth is reachable.
- **Being near the ceiling means there is little left to win.** vLLM buys only
  1.19x on this card (see `baseline-vllm-results-rtx4060ti.md`), against 4.97x on
  the slow-host box. An engine can only return the overhead that was there.
- **Prefill is CPU-insensitive.** 6% apart across two boxes whose decode differs
  by 2.3x.
- **The baseline is reproducible to 0.1%** across two runs three days apart,
  which is what makes it usable as the canonical card.

## What carries into the next tasks

- This card is the canonical one for the writeup: HF baseline, vLLM baseline,
  OOM sweep, prefill sweep, and batching sweep were all measured on it.
- 3.54 ms of host overhead is the floor for what an eager Python decode loop
  costs on a good desktop CPU. Read every other card's overhead against it.
- 2945 MiB of weights leaves about 5 GB on this 8GB card for KV and activations.
  The OOM sweep crashes at ~66k tokens of context, and the measured growth rate
  is 2.21x the analytical KV prediction (see `results/rtx4060ti/oom_curve.png`).
  That 2.21x is now explained: the KV cache itself is exactly the analytical
  size, and the excess is a per-step reallocation transient that peak memory
  captures and live memory does not. See `kv-cache-growth.md`.
