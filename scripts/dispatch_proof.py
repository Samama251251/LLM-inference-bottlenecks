"""Prove the dispatch wall by removing dispatch overhead on fixed hardware.

Every cross-card comparison in this repo varies CPU and GPU and torch build at
once, so none of them can prove on their own that the host is what limits eager
decode. This script removes the confound: one card, one box, one process, one
model, one workload. The only thing that changes is how many times Python has to
dispatch work to the GPU per token.

  eager   the baseline decode loop: back into Python every token, walk 28
          layers, dispatch 1,282 kernels (measured, see profile_decode.py).
  compiled  torch.compile(mode="reduce-overhead"), which traces the decode step
          and replays it as a CUDA graph, so one launch hands the GPU the whole
          step and the interpreter is out of the inner loop.

If decode were bandwidth-bound, both should land at the same tok/s, because the
weights have to be streamed either way and the GPU is the same piece of silicon.
Any gain is overhead that the host was adding and the graph removed.

Both modes use StaticCache, preallocated to max_cache_len. That is required for
the compiled path (CUDA graphs need static shapes, and a cache that grows by
reallocation changes shape every step), and using it for the eager path too
keeps the comparison honest: the only difference between the two arms is the
dispatch mechanism, not the cache implementation.

Reported per mode: decode tok/s, ms per token, the unavoidable weight-read time
at this card's measured bandwidth, and the host overhead left on top of it.

Usage:
    python scripts/dispatch_proof.py \
        --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --new-tokens 256 \
        --bandwidth-gb-s 402.6 --repeats 3 \
        --csv results/rtx3070/dispatch_proof.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from bench_common import bytes_to_mib, print_env


def make_cache(model, max_len: int, device: str):
    """StaticCache constructor signature moved across transformers versions."""
    cfg, dt = model.config, torch.float16
    try:
        return StaticCache(config=cfg, max_batch_size=1, max_cache_len=max_len,
                           device=device, dtype=dt)
    except TypeError:
        return StaticCache(config=cfg, batch_size=1, max_cache_len=max_len,
                           device=device, dtype=dt)


def run_once(model, decode_fn, ids, new_tokens, device, max_len):
    """One prefill plus decode, timed separately, on a fresh static cache."""
    prompt_len = ids.shape[1]
    cache = make_cache(model, max_len, device)

    with torch.inference_mode():
        pos = torch.arange(prompt_len, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=ids, past_key_values=cache,
                    cache_position=pos, use_cache=True)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0

        nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(new_tokens - 1):
            pos = torch.tensor([prompt_len + i], device=device)
            out = decode_fn(input_ids=nxt, past_key_values=cache,
                            cache_position=pos, use_cache=True)
            nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0

    return prefill_s, decode_s, new_tokens - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--bandwidth-gb-s", type=float, required=True,
                    help="measured achievable bandwidth, the MBU denominator")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print_env()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(args.device).eval()

    weights_b = torch.cuda.memory_allocated()
    weights_gb = weights_b / 1e9
    floor_ms = weights_gb / args.bandwidth_gb_s * 1000
    print(f"\nweights {bytes_to_mib(weights_b):.0f} MiB = {weights_gb:.3f} GB")
    print(f"weight read per token at {args.bandwidth_gb_s:.1f} GB/s = "
          f"{floor_ms:.2f} ms  (ceiling {1000/floor_ms:.1f} tok/s)\n")

    ids = torch.randint(1000, 5000, (1, args.prompt_tokens), device=args.device)
    max_len = args.prompt_tokens + args.new_tokens + 8

    modes = {"eager": model,
             "compiled": torch.compile(model, mode="reduce-overhead")}

    rows, summary = [], {}
    for name, fn in modes.items():
        print(f"--- {name}: warmup (compile and graph capture happen here) ---")
        run_once(model, fn, ids, min(args.new_tokens, 32), args.device, max_len)
        run_once(model, fn, ids, min(args.new_tokens, 32), args.device, max_len)

        tps = []
        for r in range(args.repeats):
            pre_s, dec_s, n = run_once(model, fn, ids, args.new_tokens,
                                       args.device, max_len)
            rate = n / dec_s
            tps.append(rate)
            ms = 1000 / rate
            print(f"  run {r+1}: prefill {pre_s*1000:6.1f} ms | "
                  f"decode {rate:6.1f} tok/s | {ms:5.2f} ms/token | "
                  f"MBU {rate*weights_gb/args.bandwidth_gb_s*100:4.1f}% | "
                  f"host overhead {ms-floor_ms:5.2f} ms")
            rows.append({
                "mode": name, "repeat": r, "prompt_tokens": args.prompt_tokens,
                "new_tokens": n, "prefill_seconds": pre_s, "decode_seconds": dec_s,
                "decode_tokens_per_sec": rate,
                "ms_per_token": ms,
                "weight_read_ms": floor_ms,
                "host_overhead_ms": ms - floor_ms,
                "mbu_pct": rate * weights_gb / args.bandwidth_gb_s * 100,
                "bandwidth_gb_s": args.bandwidth_gb_s,
                "gpu_name": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
            })
        summary[name] = statistics.median(tps)
        print()

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    e, c = summary["eager"], summary["compiled"]
    eo, co = 1000/e - floor_ms, 1000/c - floor_ms
    print("=" * 66)
    print(f"{'':<12}{'tok/s':>9}{'ms/tok':>9}{'MBU':>8}{'host overhead':>15}")
    for name, rate in (("eager", e), ("compiled", c)):
        print(f"{name:<12}{rate:>9.1f}{1000/rate:>9.2f}"
              f"{rate*weights_gb/args.bandwidth_gb_s*100:>7.1f}%"
              f"{1000/rate-floor_ms:>14.2f} ms")
    print("=" * 66)
    print(f"speedup {c/e:.2f}x on identical hardware, same process.")
    print(f"host overhead {eo:.2f} -> {co:.2f} ms/token "
          f"({(1-co/eo)*100:.0f}% of it removed).")
    print("The GPU did not change. Anything gained here was the host.")
    print(f"rows -> {args.csv}")


if __name__ == "__main__":
    main()
