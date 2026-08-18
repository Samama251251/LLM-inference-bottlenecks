"""Find out why measured KV growth is 2.21x the analytical prediction.

The OOM sweep on two independent cards measured peak allocated memory growing at
63,4xx bytes per token of context, against an analytical KV-cache cost of
2 x layers x kv_heads x head_dim x dtype_bytes = 28,672 bytes per token for
Qwen2.5-1.5B. Both cards agreed to three significant figures, so the gap is
systematic in the HuggingFace decode path rather than noise or a card artifact.

This script decomposes that. At checkpoints during a normal decode loop it splits
the live GPU allocation into:

  1. the analytical KV prediction,
  2. the bytes actually held by the cache tensors, walked directly out of
     past_key_values (so dtype and padding are observed, not assumed),
  3. total torch allocation,
  4. the residual, meaning everything that is neither weights nor cache tensors.

Those four numbers separate the candidate explanations cleanly:

  - if (2) is ~2.2x (1), the cache itself is bigger than the formula says, and
    the shapes and dtype printed at the first checkpoint say why (an fp32 cache,
    a padded head_dim, or GQA heads stored expanded to the query count),
  - if (2) matches (1) but (4) grows linearly, the cache is exactly the predicted
    size and something else scales with context. The prime suspect is cache
    growth by reallocation: torch.cat allocates a new tensor for the whole
    sequence while the old one is still live,
  - if neither grows, the gap is allocator behaviour rather than live bytes, and
    the reserved-versus-allocated split in the OOM CSVs is the place to look.

Usage:
    python scripts/kv_growth_probe.py \
        --model Qwen/Qwen2.5-1.5B \
        --checkpoint-every 500 --max-tokens 4000 \
        --csv results/rtx3070/kv_growth_probe.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_common import bytes_to_mib, kv_cache_bytes, print_env


def cache_tensors(past):
    """Yield every tensor in the KV cache, across the transformers versions that
    return a Cache object and the older ones that return tuples of tuples."""
    if past is None:
        return
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        for t in list(past.key_cache) + list(past.value_cache):
            if torch.is_tensor(t):
                yield t
        return
    for layer in past:
        for t in layer:
            if torch.is_tensor(t):
                yield t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--csv", required=True)
    ap.add_argument(
        "--cache-impl", choices=["legacy", "dynamic"], default="legacy",
        help="legacy passes past_key_values as the tuple-of-tuples the old "
             "transformers API returns, which is what baseline_hf.py and "
             "oom_sweep.py do. dynamic passes a DynamicCache object, which "
             "updates in place instead of rebuilding the tuple each step.",
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print_env()
    torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(args.device).eval()

    cfg = model.config
    L = cfg.num_hidden_layers
    KVH = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    HD = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    per_token = kv_cache_bytes(L, KVH, HD, 1)
    weights = torch.cuda.memory_allocated()
    print(f"\nconfig: {L} layers, {KVH} kv heads, head_dim {HD}, "
          f"{cfg.num_attention_heads} query heads")
    print(f"analytical KV: {per_token:,} B/token")
    print(f"weights resident: {bytes_to_mib(weights):.0f} MiB\n")

    ids = torch.randint(1000, 5000, (1, args.prompt_tokens), device=args.device)
    print(f"cache impl: {args.cache_impl}")
    rows = []
    printed_shapes = False

    with torch.inference_mode():
        if args.cache_impl == "dynamic":
            from transformers import DynamicCache
            out = model(input_ids=ids, use_cache=True,
                        past_key_values=DynamicCache())
        else:
            out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        generated = 0

        while generated < args.max_tokens:
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            generated += 1

            if generated % args.checkpoint_every:
                continue

            seq = args.prompt_tokens + generated
            tensors = list(cache_tensors(past))
            actual_kv = sum(t.numel() * t.element_size() for t in tensors)
            total = torch.cuda.memory_allocated()
            predicted = kv_cache_bytes(L, KVH, HD, seq)
            residual = total - weights - actual_kv

            if not printed_shapes and tensors:
                t = tensors[0]
                print(f"cache tensor 0: shape {tuple(t.shape)}, dtype {t.dtype}, "
                      f"{t.numel()*t.element_size():,} B  ({len(tensors)} tensors "
                      f"= {L} layers x 2)")
                printed_shapes = True

            rows.append({
                "seq_len": seq,
                "predicted_kv_b": predicted,
                "actual_kv_b": actual_kv,
                "total_allocated_b": total,
                "weights_b": weights,
                "residual_b": residual,
                "peak_allocated_b": torch.cuda.max_memory_allocated(),
                "reserved_b": torch.cuda.memory_reserved(),
            })
            print(f"seq {seq:>6} | predicted KV {bytes_to_mib(predicted):>7.1f} "
                  f"| actual KV {bytes_to_mib(actual_kv):>7.1f} "
                  f"| ratio {actual_kv/predicted:>5.2f}x "
                  f"| residual {bytes_to_mib(residual):>7.1f} "
                  f"| total {bytes_to_mib(total):>8.1f} MiB")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def slope(key):
        xs = [r["seq_len"] for r in rows]
        ys = [r[key] for r in rows]
        n = len(xs); sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
        return (n * sxy - sx * sy) / (n * sxx - sx * sx)

    print(f"\n{'':<22}{'B/token':>12}{'vs analytical':>15}")
    for label, key in (("analytical KV", "predicted_kv_b"),
                       ("actual cache tensors", "actual_kv_b"),
                       ("total allocated", "total_allocated_b"),
                       ("residual (non-cache)", "residual_b")):
        s = slope(key)
        print(f"{label:<22}{s:>12,.0f}{s/per_token:>14.2f}x")
    peak_slope = slope("peak_allocated_b")
    live_slope = slope("actual_kv_b")
    print(f"\ntransient carried by peak but not by live memory: "
          f"{(peak_slope - live_slope):,.0f} B/token "
          f"({(peak_slope - live_slope)/live_slope:.2f}x the cache itself)")
    print(f"OOM sweeps measured 63,4xx B/token of PEAK growth on two other cards.")
    print(f"rows -> {args.csv}")


if __name__ == "__main__":
    main()
