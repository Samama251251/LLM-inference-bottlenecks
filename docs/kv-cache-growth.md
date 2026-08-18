# Why the OOM curve grew 2.21x faster than the KV-cache formula

> Measured on the Vast.ai RTX 3070 8GB box (AMD Ryzen 7 3700X host), torch
> 2.11.0+cu128, transformers 4.46.3. Raw rows: `results/rtx3070/kv_growth_legacy.csv`
> and `results/rtx3070/kv_growth_dynamic.csv`. Figure: `results/rtx3070/kv_growth.png`.
> Produced by `scripts/kv_growth_probe.py`, plotted by `scripts/plot_kv_growth.py`.

The OOM sweeps left an open question. Peak allocated memory grew at 63,452 bytes
per token of context on the RTX 3060 and 63,411 on the RTX 4060 Ti, against an
analytical KV-cache cost of 28,672 bytes per token. Both cards landed on 2.21x
and agreed to three significant figures, so it was systematic rather than noise,
and nothing in the repo explained it.

The answer: the KV cache is exactly the size the formula predicts. The excess is
a per-step transient that `max_memory_allocated()` records and
`memory_allocated()` does not, and how large it is depends on which cache
implementation transformers uses.

## The formula was never wrong

`scripts/kv_growth_probe.py` walks the tensors in `past_key_values` during a
normal decode loop and compares their real byte count against the prediction, at
ten checkpoints from 2,032 to 20,032 tokens of context.

The first cache tensor is shaped `(1, 2, 532, 128)` with dtype `torch.float16`,
and there are 56 of them, 28 layers times K and V. That rules out the obvious
suspects in one line: the dtype is fp16 and not fp32, the 2 KV heads are stored
collapsed rather than expanded to the 12 query heads, and `head_dim` is 128 with
no padding.

| series | B/token | vs analytical |
| --- | --- | --- |
| analytical KV | 28,672 | 1.00x |
| live cache tensors | 28,672 | **1.00x** |
| live total allocated | 28,709 | 1.00x |
| residual (neither weights nor cache) | 37 | 0.00x |

The ratio of measured cache bytes to predicted is exactly 1.00 at all ten
checkpoints, not on average. Live total allocation grows at 28,709 B/token, which
is the cache plus 37 bytes per token of everything else. The memory a decode loop
actually needs is the formula, full stop.

## What peak was measuring

Peak is a different quantity, and it is the one the OOM sweep recorded.

| cache implementation | peak growth | vs analytical | transient |
| --- | --- | --- | --- |
| legacy tuple-of-tuples | 92,317 B/token | 3.22x | 2.22x the cache |
| `DynamicCache` object | 63,595 B/token | 2.22x | 1.22x the cache |

Switching from the legacy tuple API to a `DynamicCache` object removes exactly
one full copy of the cache from the peak, 3.22x down to 2.22x. Nothing else
changed: same model, same prompt, same decode loop, same generated tokens.

The mechanism is cache growth by reallocation. Appending a token means building a
new tensor that holds the whole sequence while the old one is still referenced,
so both are live at once. The legacy path keeps the previous cache tuple alive
across the whole forward pass, so it pays for an extra complete copy that the
in-place path does not.

The `DynamicCache` figure of 63,595 B/token matches the 63,452 and 63,411
measured on the 3060 and 4060 Ti to within 0.3%. Those sweeps were already on the
efficient path, and the 1.22x transient is what that path costs.

## What it costs in context length

On an 8 GiB card with 2,945 MiB of weights resident, dividing the remaining
memory by each peak growth rate:

```
legacy tuple    59,595 tokens
DynamicCache    86,509 tokens     1.45x more context
```

The cache implementation, which changes no arithmetic and no output token,
decides how much context fits before CUDA runs out of memory.

## Reading the OOM curves correctly

Two consequences for `results/rtx3060/oom_curve.png` and
`results/rtx4060ti/oom_curve.png`:

The blue analytical line on those figures is not only a prediction. It is also,
to within 0.13%, where live memory actually sits. The red measured line above it
is peak, carrying the transient.

The crash is triggered by reserved memory, not by either of those. At OOM the
3060 held 10,437 MiB allocated against 11,764 reserved of 12,288, and the 4060 Ti
6,964 against 7,772 of 8,192. Roughly 0.8 to 1.3 GB of allocator fragmentation
sits between what is live and what the driver has handed out, and it is the
reserved figure that has to fit in the card.

## Caveats

The A/B ran on one card, one model, one prompt length, batch 1, single run per
arm with no repeats. Memory figures are deterministic in a way timing is not, so
repeats matter less here, but the transient multipliers are one measurement each.

The 3.22x legacy figure is specific to transformers 4.46.3 on torch 2.11.0+cu128.
The 3060 and 4060 Ti sweeps ran on torch 2.12.0 and 2.11.0 with different CUDA
builds and produced 2.21x, matching the dynamic arm here rather than the legacy
one. So the exact transient depends on the version combination, and the invariant
worth carrying forward is that live memory equals the formula while peak does not.

I did not verify the reallocation mechanism directly by inspecting allocator
events. The 1.00x-exactly cache measurement and the clean one-copy difference
between the two arms are consistent with it, but the mechanism is inference from
those two numbers rather than an observation.
