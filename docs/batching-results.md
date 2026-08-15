# Batching sweep (RTX 4060 Ti, vLLM): where throughput stops being free

> **Hardware: NVIDIA RTX 4060 Ti 8GB (Ada AD106, sm_89), vLLM 0.24.0.** Measured
> on the Vast.ai 4060 Ti box, since destroyed. Raw rows:
> `results/rtx4060ti/vllm_batch_sweep.csv`. Figure:
> `results/rtx4060ti/batch_curve.png`. Produced by
> `scripts/bench_vllm_batch.py`, plotted by `scripts/plot_batch_sweep.py`.

Everything measured before this is batch 1, which is latency, not throughput. A
serving engine's real job is many sequences at once. This sweep sends a growing
number of concurrent sequences to vLLM and watches aggregate throughput and
per-sequence latency until the KV pool runs out.

Two pool configurations were run: `realistic` at `gpu_memory_utilization=0.9`
(6218 KV blocks, room for 129 sequences) and `constrained` at 0.65 (1759 blocks,
36 sequences), so the KV wall could be observed twice, at a cheap batch size and
an expensive one. Two repeats per point, batch 1 to 256.

## Measured (realistic pool, averaged over 2 repeats)

| batch | decode tok/s | scaling | % of ideal | TPOT ms | TTFT s | queued |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 83.3 | 1.0x | 100% | 12.01 | 0.04 | |
| 2 | 164.2 | 2.0x | 99% | 12.18 | 0.07 | |
| 4 | 322.4 | 3.9x | 97% | 12.41 | 0.14 | |
| 8 | 623.9 | 7.5x | 94% | 12.82 | 0.28 | |
| 16 | 1189.2 | 14.3x | 89% | 13.45 | 0.56 | |
| 32 | 2155.2 | 25.9x | 81% | 14.85 | 1.11 | |
| 64 | 3615.7 | 43.4x | 68% | 17.70 | 2.21 | |
| 128 | **5353.7** | 64.3x | 50% | 23.91 | 4.43 | |
| 192 | 3435.7 | 41.3x | 21% | 55.88 | 6.64 | **yes** |
| 256 | 4103.8 | 49.3x | 19% | 62.38 | 8.86 | **yes** |

Constrained pool: identical through batch 32, then queues at 48 (1749.9 tok/s,
TPOT 27.43 ms) and 64 (1962.2 tok/s, TPOT 32.62 ms).

The headline for serving: **5354 tok/s at batch 128 against 70.1 tok/s for the HF
baseline at batch 1, a 76x difference.** The batch-1 vLLM comparison in
`baseline-vllm-results-rtx4060ti.md` is 1.19x. Almost all of a serving engine's
advantage is in batching, not in single-stream decode.

## The free lunch, and why it exists

Through batch 16 throughput scales at 89% of ideal while TPOT moves from 12.01 to
13.45 ms. Sixteen times the work for 12% more latency per token.

The reason is that decode at batch 1 is memory-bound on the weights. Every step
streams all 3.088 GB of fp16 weights regardless of how many sequences are riding
along, so the second sequence, and the sixteenth, reuse weight traffic that was
already being paid for. Extra sequences are nearly free until something else
starts to cost.

## What actually ends the free lunch (not what we assumed)

The obvious guess is that decode goes compute-bound once the batch is large
enough to fill the GEMMs. `bench_vllm_batch.py` says as much in its docstring.
**The data does not support it.** Accounting for both memory traffic and FLOPs
per decode step, with `avg context = 512 prompt + 128 mean generated = 640`:

| batch | weight GB | KV GB | total GB | GB/s | % of 288 | TFLOPS | % of 88 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.088 | 0.018 | 3.107 | 259 | 90% | 0.3 | 0% |
| 8 | 3.088 | 0.147 | 3.235 | 252 | 88% | 1.9 | 2% |
| 32 | 3.088 | 0.587 | 3.676 | 248 | 86% | 6.7 | 8% |
| 64 | 3.088 | 1.174 | 4.263 | 241 | 84% | 11.2 | 13% |
| 128 | 3.088 | 2.349 | 5.437 | 227 | 79% | 16.5 | 19% |
| 192 | 3.088 | 3.523 | 6.612 | 118 | 41% | 10.6 | 12% |

Compute never exceeds 19% of the tensor-core peak. Decode never goes
compute-bound in this sweep at all. The machine stays memory-bound from batch 1
to batch 128, holding 79% to 90% of peak bandwidth throughout.

What changes is **what the bandwidth is being spent on**. At batch 1, 99% of the
traffic is weights and the KV cache is a rounding error. At batch 128 the KV
cache is 2.35 GB per step against 3.088 GB of weights, so 43% of all memory
traffic is now KV reads. Attention has to read every sequence's entire cache on
every step, and that grows with batch and with context while the weight read
stays constant.

So the free lunch is not ended by running out of compute. It is ended by the KV
cache becoming a first-class consumer of the same bandwidth the weights were
using. Scaling efficiency falling from 89% at batch 16 to 50% at batch 128 tracks
the KV share of traffic rising, not any compute ceiling being approached.

This is worth stating plainly because it changes the conclusion. If decode were
compute-bound at high batch, more FLOPs would help. It is not, so they would not.
The lever is KV traffic: shorter contexts, fewer KV heads, quantized or
compressed KV. That is also why PagedAttention matters beyond memory tidiness.

## The KV wall

Past the pool's capacity the scheduler cannot run every sequence in one wave. It
runs a subset and defers the rest, `queued` flips true, and throughput does not
merely flatten, it **drops**: 5354 tok/s at batch 128 falls to 3436 at 192.
Latency roughly doubles at the same point, TPOT going from 23.91 to 55.88 ms,
because a deferred sequence waits a full wave before its next token.

Bus utilization collapses to 41% at batch 192. Past the wall the hardware is not
merely saturated, it is idle part of the time while the scheduler shuffles work.

vLLM's own block accounting predicts both walls closely:

| pool | gpu_mem_util | KV blocks | capacity (seqs) | first queued |
| --- | --- | --- | --- | --- |
| realistic | 0.9 | 6218 | 129 | batch 192 |
| constrained | 0.65 | 1759 | 36 | batch 48 |

In both cases the first swept batch size above `kv_capacity_seqs` is the first
one that queues. The engine tells you where its wall is before you hit it.

One correction to the original script docstring, which expected the realistic
pool to hold "hundreds of sequences" so that only the constrained run would show
the wall. It holds 129, and the realistic run walks straight into the wall at 192
on its own. The constrained run is still useful, since it puts the wall at a
cheap batch size and confirms the capacity number predicts it, but it was not
necessary to see the effect.

## The wall is queuing, not bytes

Device VRAM moves from 7557 to 7910 MiB across the entire realistic sweep, a
change of under 5%, while throughput rises 64x and then collapses. vLLM reserves
its KV pool at startup, so you cannot watch memory fill up the way the HF OOM
sweep does in `results/rtx4060ti/oom_curve.png`. The same wall shows up as a
crash in one engine and as queued requests in the other.

That contrast is the clearest practical statement of what PagedAttention buys.
Same card, same model, same finite KV capacity. HF fragments its cache, grows it
by reallocation, and dies with a CUDA OOM at ~66k tokens. vLLM pages the cache
into fixed blocks, knows exactly how many sequences it can hold, and degrades by
scheduling instead of crashing.

## Caveats

- TPOT (decode wall / decode steps) is the primary latency measure. Per-request
  p50 and p95 from `RequestOutput.metrics` were not populated on this vLLM build
  and are NaN in the CSV.
- The traffic and FLOP table above is derived from the measured TPOT plus the
  model's architecture, not from a profiler. It is an accounting argument, and a
  profiled run would make it a measurement.
- Prompt length is fixed at 512 and `max_model_len` at 784. KV traffic grows with
  context, so a longer-context sweep would end the free lunch sooner.
- Single card, single model. The 79% to 90% bandwidth figures are specific to a
  1.5B model on a 288 GB/s card.

## Key learnings

- **Batching, not batch-1 decode, is where a serving engine wins.** 1.19x at
  batch 1, 76x at batch 128 against the HF baseline.
- **Throughput is nearly free to batch 16** (89% of ideal, 12% latency cost),
  because extra sequences ride along on weight traffic already being paid for.
- **Decode never goes compute-bound here**, peaking at 19% of tensor peak. The
  free lunch ends because KV reads grow into the same bandwidth budget, reaching
  43% of all traffic at batch 128.
- **The KV wall makes throughput fall, not flatten**, and vLLM's
  `kv_capacity_seqs` predicts where it lands in both pool configurations.
- **With a reserved pool the wall is queuing, not bytes.** Device VRAM is flat
  across the whole sweep.

## What carries into the next tasks

- The KV-traffic accounting is the bridge to Phase 2. Optimizations that reduce
  KV bytes read per step (quantized KV, fewer KV heads, shorter effective
  context) attack the actual limiter; more FLOPs do not.
- Peak useful batch on this card and model is ~128, at 50% scaling efficiency and
  24 ms TPOT. That is the throughput/latency knob a real deployment tunes.
- The HF-crashes versus vLLM-queues contrast is the concrete payoff of
  PagedAttention and belongs in the writeup next to the OOM curve.
