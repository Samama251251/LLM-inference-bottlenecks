# Moving to a faster card made decoding slower

Decoding one token at a time is limited by memory bandwidth, so I expected a card
with more bandwidth to decode faster. It did not: an RTX 3070 Ti with 608 GB/s
decoded Qwen2.5-1.5B at 30.56 tokens per second while an RTX 4060 Ti with 288
GB/s reached 70.09, and on a single machine with the GPU held fixed I later
recovered 1.687x by cutting the work the CPU does between tokens, with a profiler
showing the card idle for 41% of every token.

## The thing that did not make sense

I have run the same script on four rented single-GPU boxes. Same model, same
prompt length, same number of generated tokens, same batch size of 1, plain
HuggingFace `transformers` in eager mode each time.

| card | spec bandwidth | HF decode | runs | host CPU | MBU |
| --- | --- | --- | --- | --- | --- |
| RTX 4060 Ti | 288 GB/s | 70.09 tok/s | 2 | i7-11700 | 75.2% |
| RTX 3070 | 448 GB/s | 59.75 tok/s | 5 | Ryzen 7 3700X | 41.2% |
| RTX 3070 Ti | 608 GB/s | 30.56 tok/s | 1 | Xeon E5-2680 v4 | 15.5% |
| RTX 3060 | 360 GB/s | 20.59 tok/s | 2 | not recorded | 17.7% |

MBU is the share of the card's bandwidth the loop actually used: tokens per
second times 3.088 GB of weights per token, over spec bandwidth.

Reading down the bandwidth column and then the throughput column, the order is
close to reversed. The 3070 Ti has 2.11x the bandwidth of the 4060 Ti and 0.44x
the throughput. What does line up is the CPU: the newest host is fastest, the
2016 Xeon is slowest, and the 2019 Ryzen sits between them.

Three caveats before that table is taken too seriously. The 3070 Ti figure is a
single run with no repeat, so I have no spread for it. The 3060's two runs were
18.0 and 23.18 tokens per second, 29% apart, so its median is soft. And the CPU
column only comes from the result file for the RTX 3070 row; I only added a CPU
field to the logging code during this session, so the other three CPUs come from
notes I wrote when I rented those machines. Each row also changes the CPU, the
GPU, and the torch build together. On its own this table is something odd, not a
demonstration of cause.

So I rented one box and tried to test the idea properly.

## Setup

One NVIDIA RTX 3070, 8192 MiB, compute capability 8.6, 46 SMs, rented on
Vast.ai. PyTorch reports 7840 MiB of that as usable. Host is an AMD Ryzen 7
3700X with 16 logical cores, and the listing allocated 5.3 of 16 to my instance,
which matters later.

Model is Qwen2.5-1.5B in fp16: 28 layers, 12 query heads, 2 key/value heads,
head dimension 128. Weights sit at 2945.29 MiB resident, or 3.088 GB, and that
came out byte-identical on all four cards, which is how I check the model loaded
the same way everywhere.

Two Python environments, because vLLM pins its own torch and installing it
alongside the baseline would replace the torch the baseline was measured on. The
HuggingFace side ran torch 2.11.0+cu128 with transformers 4.46.3. vLLM 0.24.0
ran in a separate venv on torch 2.11.0+cu130.

Every run used a 512-token prompt and generated 255 tokens greedily at batch 1.
Prefill and decode are timed separately, never blended. I call
`torch.cuda.synchronize()` before reading any timer, discard a warmup generation,
and reset peak memory counters before each measured run.

## Finding the real ceiling first

The whole argument is a ratio, so I measured the denominator instead of trusting
it.

The RTX 3070 is specified at 448 GB/s. The Vast.ai dashboard reported 385.8
GB/s, and earlier in the same session it had reported 179.6 GB/s for the same
machine, which told me the dashboard figure moves with host load. So I measured
it: a large contiguous fp16 copy, best of 20 runs, gave 402.6 GB/s of combined
read and write traffic, 90% of spec.

At 402.6 GB/s, streaming 3.088 GB of weights takes 7.671 ms, so nothing can
decode faster than 130.4 tokens per second at batch 1 on this card.

Measured HuggingFace decode was 59.75 tokens per second, median of five runs
(55.07, 59.76, 58.03, 60.25, 59.75). That is 45.8% of the ceiling. Per token it
spends 16.74 ms where only 7.671 ms is unavoidable, leaving 9.07 ms of something
else.

The five runs span 9.4%, the first the slowest at 55.07 and the rest between
58.03 and 60.25. I ran five rather than two because I shared the CPU with other
tenants. vLLM on the same box gave 115.96, 115.57 and 115.84, a spread of 0.34%,
so the noise sits in the eager path, not the machine.

## Removing the CPU work, hardware unchanged

The eager decode loop goes back into Python for every token and walks all 28
layers, sending the GPU a long series of small jobs. `torch.compile` with
`mode="reduce-overhead"` traces that step once and replays it as a CUDA graph, so
a single call hands the GPU the whole step.

Both arms used a preallocated `StaticCache`. CUDA graphs need shapes that do not
change, and a cache that grows by reallocation changes shape every token, so the
compiled arm required it. I used it in the eager arm too so the only difference
between the two was how work reaches the GPU.

| arm | decode | ms/token | MBU | leftover per token |
| --- | --- | --- | --- | --- |
| eager | 54.33 tok/s | 18.41 | 41.68% | 10.73 ms |
| compiled | 91.65 tok/s | 10.91 | 70.30% | 3.24 ms |

1.687x, same card, same process, same model, same cache. Prefill moved from
65.36 ms to 66.97 ms, so the compiled version was slightly slower there. That is
what I would predict if the gain came from removing per-token CPU work: prefill
is one large parallel pass that already keeps the GPU busy, so there is no gap to
close.

![Per-token decode time on the RTX 3070, split into the unavoidable weight read,
other GPU work, and time the GPU spent idle. Bars are ms per decoded token.
The GPU-busy figure was profiled on the eager path only and reused for the
compiled bar, which is an assumption. vLLM was never profiled, so its bar shows
total wall time with the split left blank.](../results/rtx3070/decode_budget.png)

## A second measurement of the same thing

I profiled twelve steady-state decode steps with `torch.profiler`, counting only
device-side kernels.

One decode token launches 1282 CUDA kernels. Total GPU time across them is 10.806
ms, so the average kernel runs for 8.429 microseconds. The largest single
contributor is a `gemv2T_kernel_val` half-precision kernel, 57 launches per
token totalling 5.019 ms.

Put that next to the wall-clock numbers above:

| arm | wall | GPU busy | GPU idle |
| --- | --- | --- | --- |
| eager | 18.41 ms | 10.806 ms | 7.60 ms (41%) |
| compiled | 10.91 ms | 10.806 ms | 0.10 ms (1%) |

The compiled wall time lands within 0.1 ms of the GPU's own busy time. Two
methods that share no code agree: eager decoding leaves the card waiting 41% of
each token, and replaying the step as a graph closes that to roughly nothing.

Documents in this repo previously said "on the order of 500 kernels per token."
That was a guess I made from the layer count. The measured number is 1282, and I
have corrected it.

## What the OOM curve was really showing

A separate question was open from earlier runs. Pushing context length until the
card died, peak memory grew at 63,452 bytes per token on the RTX 3060 and 63,411
on the 4060 Ti, against an analytical KV-cache cost of 2 x 28 layers x 2 KV heads
x 128 x 2 bytes = 28,672 bytes per token. Both cards agreed on 2.21x and I had no
explanation.

I wrote a probe that walks the cache tensors directly during decoding. The first
tensor is shaped (1, 2, 532, 128) and is `torch.float16`, so no expanded heads,
no padding, no accidental fp32. Across ten checkpoints from 2032 to 20032 tokens,
the ratio of measured cache bytes to predicted is exactly 1.00 at every one. Live
total allocation grows at 28,709 bytes per token, and everything that is not
weights or cache adds 37 bytes per token.

The formula was right. What the OOM curve tracked was peak, and peak carries a
transient: the old cache is still alive while the next one is being built.

Swapping the legacy tuple cache for a `DynamicCache` object changed peak growth
from 92,317 to 63,595 bytes per token, which is exactly one full copy of the
cache removed. On an 8 GiB card that is 59,595 tokens of context versus 86,509,
1.45x more, from a change that alters no arithmetic and no output.

The dynamic figure of 63,595 also matches the 63,452 and 63,411 from the other
two cards to within 0.3%, which suggests those sweeps were already on the
efficient path and the remaining 1.22x transient is what that path costs.

![Memory against context length on the RTX 3070, in MiB. Live allocation sits on
the analytical KV line at 1.00x. Peak allocation runs 3.22x higher with the
legacy tuple cache and 2.22x with DynamicCache. Ten checkpoints per arm from
2032 to 20032 tokens, single run each, no repeats.](../results/rtx3070/kv_growth.png)

## Three things that came out wrong

`StaticCache` is not free. Eager decoding with it ran at 54.33 tokens per second
against 59.75 with the normal cache, about 9% slower. So 1.687x is the speedup
within a controlled comparison, and against the fastest eager configuration I
measured the honest number is 91.65 / 59.75 = 1.53x. I report both.

My first profiler build summed GPU time over every event that had any, which
counted `aten::linear` wrapping `aten::matmul` wrapping `aten::mm` wrapping the
kernel that actually ran. It reported 39.67 ms of GPU work inside a 7.37 ms step
and a negative idle time. I filtered to device-side events only and measured wall
clock directly instead of reconstructing it from nested spans.

vLLM reached 115.84 tokens per second, 8.63 ms per token. That is below the
10.806 ms of GPU time the profiler measured for the eager path, so vLLM is not
only launching fewer kernels, it is running cheaper ones. I had been attributing
the whole HuggingFace-to-vLLM gap to launch overhead. At least part of it is
fused kernels doing the same math with less work, and my measurements do not
separate those two effects.

## What this does not show

The four-card table cannot establish that the CPU caused the inversion. Each row
changes CPU, GPU, torch version and CUDA build together, one row is a single run,
and one card's CPU was never recorded. The controlled result is the single-box
comparison, and it shows only that per-token CPU work cost 7.60 ms on this
machine.

MBU denominators are inconsistent across the table. I measured achievable
bandwidth only on the RTX 3070, at 402.6 GB/s. The other three rows divide by
vendor spec, which will understate their MBU by roughly the 10% gap I measured
here.

Everything is batch 1, one model, one prompt length. Batch 1 is where launch
overhead matters most, because the GPU work per kernel is smallest. None of this
says anything about throughput under load.

The 5.3 of 16 vCPU allocation means another tenant's work could have moved my
eager numbers, and the 9.4% spread is consistent with that. The vLLM and compiled
paths were far more stable, which is what I would expect if the noise enters
through host-side work, though I did not test that directly.
