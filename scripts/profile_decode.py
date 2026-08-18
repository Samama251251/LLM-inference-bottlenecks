"""Count the kernels in one decode token, and measure how long the GPU idles.

Every writeup in this repo says eager decode dispatches "on the order of 500
kernels per token" and that the GPU sits idle waiting on Python. Both were
estimates, derived from the layer count and inferred from MBU. This measures
them.

Uses torch.profiler, which is a userspace profiler. Vast.ai instances are
unprivileged containers with no perf or eBPF access, so kernel-level profilers
are unavailable; torch.profiler needs none of that because it hooks the CUDA
runtime through CUPTI from inside the process.

Reported per decode step:

  kernels          how many distinct CUDA kernel launches one token costs
  gpu busy         summed kernel time on the device
  wall             wall-clock time for the step
  gpu idle         wall minus busy, the time the card spent waiting on the host
  per-kernel       average GPU time per kernel, which is what makes the launch
                   overhead matter: when a kernel runs for a few microseconds,
                   the cost of launching it is no longer negligible beside it

A high idle fraction is the dispatch wall stated directly, without going through
MBU as a proxy. Run it next to dispatch_proof.py: that one shows removing the
dispatch overhead makes decode faster, this one shows the overhead is there.

Usage:
    python scripts/profile_decode.py \
        --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --steps 12 \
        --json results/rtx3070/decode_profile.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM

from bench_common import print_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--steps", type=int, default=12,
                    help="decode steps to profile; averaged over")
    ap.add_argument("--json", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print_env()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    cfg = model.config
    print(f"\n{cfg.num_hidden_layers} layers\n")

    ids = torch.randint(1000, 5000, (1, args.prompt_tokens), device=args.device)

    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)

        # Warm the loop so we profile steady state, not first-call setup.
        for _ in range(8):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)

        torch.cuda.synchronize()
        wall_t0 = __import__("time").perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.steps):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
        wall_s = __import__("time").perf_counter() - wall_t0

    events = prof.key_averages()
    # Count ONLY device-side kernels. An aten:: entry is a CPU-side operator
    # wrapper that also carries device_time, so summing everything with
    # device_time > 0 counts the same GPU work several times over: aten::linear
    # wraps aten::matmul wraps aten::mm wraps the cutlass kernel that actually
    # runs. Filtering on DeviceType.CUDA keeps just the real launches.
    kernels = [e for e in events if e.device_type == DeviceType.CUDA]
    n_launch = sum(e.count for e in kernels)
    gpu_us = sum(e.device_time_total for e in kernels)

    # Wall clock is measured directly around the profiled loop rather than
    # reconstructed from event spans, which nest and cannot simply be summed.
    total_wall_us = wall_s * 1e6

    steps = args.steps
    per_step_kernels = n_launch / steps
    per_step_gpu_ms = gpu_us / steps / 1000
    per_step_wall_ms = total_wall_us / steps / 1000
    idle_ms = per_step_wall_ms - per_step_gpu_ms

    print("=" * 60)
    print(f"{'kernels per decode token':<32}{per_step_kernels:>12,.0f}")
    print(f"{'GPU busy per token':<32}{per_step_gpu_ms:>10.2f} ms")
    print(f"{'wall per token':<32}{per_step_wall_ms:>10.2f} ms")
    print(f"{'GPU idle per token':<32}{idle_ms:>10.2f} ms"
          f"   ({idle_ms/per_step_wall_ms*100:.0f}% of wall)")
    print(f"{'average GPU time per kernel':<32}"
          f"{per_step_gpu_ms*1000/per_step_kernels:>10.1f} us")
    print("=" * 60)
    print("note: CUPTI instrumentation inflates the wall figure well above an")
    print("unprofiled run, so the idle fraction printed here is an UPPER bound.")
    print("The kernel count and GPU-busy time are the trustworthy outputs; pair")
    print("them with an unprofiled wall (dispatch_proof.py) for the real idle.")
    print("\ntop kernels by total GPU time:")
    for e in sorted(kernels, key=lambda e: -e.device_time_total)[:12]:
        print(f"  {e.key[:52]:<54}{e.count/steps:>7.0f}/tok"
              f"{e.device_time_total/steps/1000:>9.3f} ms")

    payload = {
        "kernels_per_token": per_step_kernels,
        "gpu_busy_ms_per_token": per_step_gpu_ms,
        "wall_ms_per_token": per_step_wall_ms,
        "gpu_idle_ms_per_token": idle_ms,
        "gpu_idle_fraction": idle_ms / per_step_wall_ms,
        "avg_us_per_kernel": per_step_gpu_ms * 1000 / per_step_kernels,
        "steps_profiled": steps,
        "num_layers": cfg.num_hidden_layers,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "top_kernels": [
            {"name": e.key, "per_token": e.count / steps,
             "ms_per_token": e.device_time_total / steps / 1000}
            for e in sorted(kernels, key=lambda e: -e.device_time_total)[:20]
        ],
    }
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nprofile -> {args.json}")


if __name__ == "__main__":
    main()
