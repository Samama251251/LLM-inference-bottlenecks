# Watching the KV cache hit the wall

Phase 1 of an LLM inference optimization roadmap. Draft, written as the
measurements come in.

## The setup

One small model (Qwen2.5-1.5B, fp16), one GPU (a single RTX 3060, 12GB, Ampere
sm_86), run two ways: plain HuggingFace `transformers` as a readable control,
then vLLM as the optimized engine. Same model, same prompt (512 tokens), same 256
new tokens, greedy decoding, batch size 1. The only thing that changes between
runs is the engine, so the gap between them is attributable to the engine and
nothing else.

Everything is timed by one shared harness with a few non-negotiable rules:
prefill and decode are measured separately, the GPU is synchronized before every
timer read, a warmup run is discarded, and VRAM is logged with the peak captured.
The small card is on purpose: at 12GB the KV cache hits the wall at a realistic
context length, which is the headline experiment, not something to engineer
around.

## HuggingFace vs vLLM

| metric | HF transformers | vLLM | gap |
| --- | --- | --- | --- |
| prefill latency | ~137 ms (3740 tok/s over the prompt) | ~124 ms (4114 tok/s) | ~1.1x |
| decode tokens/sec | ~18-23 tok/s | ~79.9 tok/s | ~3.5-4x |
| decode bandwidth use | ~15-20% of 360 GB/s | ~68% | |
| VRAM | 3133 MiB peak, grown organically | 11237 MiB, reserved up front | not comparable |

Two numbers carry the whole comparison. Prefill barely moves, because both
engines hand the prompt's matrix multiplies to the same cuBLAS tensor-core
kernels, and you cannot beat cuBLAS on a raw GEMM. vLLM's ~10% edge there is
trimmed launch overhead, not a faster matmul. Decode is the opposite story: vLLM
is roughly 4x faster, and the bandwidth-use row says why. HF decode runs at only
~15-20% of the card's memory bandwidth, while vLLM reaches ~68%. The headroom was
never compute; it was that decode is memory-bound and HF was leaving most of the
memory pipe idle. The next section is why.

The VRAM column is the one trap. HF's figure is what generation organically grew
into; vLLM's 11237 MiB is a pool it reserves at startup (weights 3.0 GiB, KV pool
6.3 GiB, CUDA-graph buffers 0.4 GiB, plus context and headroom), sized to 90% of
the card. The two are not the same measurement, so the fair comparison is decode
rate and prefill latency, not peak bytes.

## I bought twice the bandwidth and decode got slower

Decode is memory-bound, so a card with more memory bandwidth should decode
faster. That is the prediction. I ran the same script on three cards to check it,
and on the card with the most bandwidth it was wrong.

| card | bandwidth | HF decode | HF MBU | vLLM decode | vLLM MBU | vLLM / HF |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 3060 12GB | 360 GB/s | 20.6 tok/s | 17.7% | 79.9 tok/s | 68.5% | 3.88x |
| RTX 4060 Ti 8GB | 288 GB/s | **70.1 tok/s** | 75.2% | 83.2 tok/s | 89.2% | 1.19x |
| RTX 3070 Ti 8GB | **608 GB/s** | 30.6 tok/s | 15.5% | **152.0 tok/s** | 77.2% | 4.97x |

Read the first two columns against each other. Ranked by bandwidth the order is
3070 Ti, 3060, 4060 Ti. Ranked by HuggingFace decode speed it is 4060 Ti, 3070
Ti, 3060. The slowest card on paper is the fastest one measured, and the 3070 Ti
has 2.11x the bandwidth of the 4060 Ti while decoding at 0.44x its speed.

Note these are the same 8GB of VRAM on both of those cards. Capacity is not the
variable here and never affects decode speed; it only sets where the KV cache
runs out. The variable is bandwidth, and more of it made things worse.

Now change the engine and nothing else. Under vLLM the ranking snaps back to
bandwidth order: 3070 Ti first, at 152 tok/s, almost exactly 2x the 4060 Ti's
bandwidth-adjusted rate. The hardware was always capable of it. The baseline
could not reach it.

MBU is the column that explains the rest. Decode has to stream every weight once
per token, which for this model in fp16 is 3.088 GB, so `tok/s = MBU x bandwidth
/ 3.088 GB`. On the 4060 Ti, HuggingFace already runs at 75% of the memory
bandwidth: the card is close to its ceiling and there is little left for a better
engine to win, which is why vLLM buys only 1.19x there. On the 3070 Ti,
HuggingFace runs at 15.5%. The 608 GB/s bus sits idle 85% of the time.

What is it waiting for? The CPU. Eager HuggingFace decode re-enters Python for
every token and walks all 28 layers, dispatching on the order of 500 separate
kernels, each one preceded by interpreter work before the GPU sees anything. Per
token that is 14.3 ms on the 4060 Ti box and 32.7 ms on the 3070 Ti box. The
difference is not the graphics cards. It is that the 4060 Ti box has an i7-11700
and the 3070 Ti box has a Xeon E5-2680 v4, a Broadwell part from 2016, and the
2.29x gap in per-token time tracks the single-thread gap between those two CPUs
far better than it tracks anything about the GPUs. The GPU finishes each tiny
kernel and then waits for Python to hand it the next one.

So there are two different walls, and "memory-bound" only names one of them.
Memory-bound is the nature of the decode workload: its ceiling is set by
bandwidth. But you only meet that ceiling if something keeps the GPU fed. Get
there through a slow interpreter on a slow host and you hit the dispatch wall
first, at 15% of the bandwidth wall, and buying a faster bus buys nothing at all.

The cleanest evidence for that is the last column, because it holds the GPU
fixed. Same card, same box, same CPU, only the engine changes. On the fast-host
4060 Ti, switching to vLLM buys 1.19x. On the slow-host Xeon box it buys 4.97x.
vLLM's batch-1 win is not a property of vLLM. It is a measurement of how much
host overhead there was to remove, which is why the same swap pays four times
better on the slower host.

Two caveats I would rather state than bury. First, the two boxes differ in CPU
and GPU and torch build at once, so the cross-card comparison on its own is
confounded; the within-box vLLM ratio is what carries the causal claim, not the
raw pair. Second, the ~500 kernels per token is an estimate from the model's
layer count, not a profiled number, so the per-kernel arithmetic above is an
order-of-magnitude argument rather than a measurement. Both are fixable with one
profiled decode step, which is the next thing to run.

The practical version, for anyone sizing a box: at batch 1 on an eager baseline,
the host CPU can be worth more than the GPU. Check your MBU before you pay for
bandwidth. If you are at 15%, a faster card will not help you and a better engine
will give you 5x on the hardware you already own.

## The OOM curve

<!-- The headline. Push context length up in steps, log peak VRAM at each step,
catch the CUDA OOM, and plot measured VRAM against the analytical KV-cache
prediction. Written the moment results/rtx4060ti/oom_curve.png exists. -->

## Why the two engines differ

The gap is a scheduling problem, not an arithmetic one. The GPU runs kernels; the
CPU, driving Python, launches them. In HuggingFace's decode loop every new token
re-enters Python and walks all 28 layers, firing hundreds of separate kernels,
each one preceded by Python interpreting the code and dispatching the op. At batch
1 the GPU work per kernel is a few microseconds, often less than the time Python
needs to launch the next one, so the GPU finishes and stalls, waiting on the CPU.
That is what ~15-20% bandwidth use means: the memory pipe is half-empty because
the bottleneck is the interpreter, not the hardware.

vLLM takes the CPU out of that loop. It captures the whole decode step into a CUDA
graph, a recording of every kernel launch in order, and replays it with a single
call, and it fuses many of those kernels into fewer, bigger ones. Now one launch
hands the GPU the entire step and it streams weights back to back with no Python
in between, which is the jump to ~68% bandwidth use. Prefill cannot benefit the
same way: it is one long parallel pass that already keeps the GPU busy, so there
is no idle gap for graphs to close, which is exactly why prefill barely moved.

None of this is PagedAttention, which is a separate win and invisible at batch 1.
PagedAttention manages the KV cache like operating-system virtual memory, paging
it into fixed blocks so it never fragments and many sequences pack into one
reserved pool. That buys high-batch throughput and long context, not single-stream
decode speed. Its payoff shows up in the next section, where the same paged pool
is what lets vLLM keep going while plain transformers fragments and hits the wall.
The throughput here, ~4x at batch 1, is the floor of vLLM's advantage, not the
ceiling: continuous batching, which this benchmark deliberately does not exercise,
is where the larger serving wins live. Those are what the later phases build.
