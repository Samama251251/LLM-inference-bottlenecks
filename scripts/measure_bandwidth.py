"""Measure achievable memory bandwidth, the denominator every MBU number uses.

MBU is measured throughput divided by bandwidth, so a wrong bandwidth figure
silently rescales every conclusion drawn from it. Three sources disagreed on this
card: the RTX 3070 is specified at 448 GB/s, the Vast.ai dashboard reported 385.8
GB/s, and earlier in the same session the same dashboard reported 179.6 GB/s for
the same machine, which showed its figure moves with host load.

So this measures it. A large contiguous fp16 copy is the traffic pattern decode
weight-streaming looks like: read everything once, write everything once, no
reuse. Best of N is reported rather than the mean, because contention on a shared
host can only make a run slower, never faster, so the fastest run is the closest
to the card's real capability.

On the RTX 3070 this measured 402.6 GB/s, 90% of the 448 GB/s spec. That is the
figure used in dispatch_proof.py and in docs/baseline-results-rtx3070.md.

Usage:
    python scripts/measure_bandwidth.py --mib 256 --iters 20
"""

from __future__ import annotations

import argparse
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=256, help="buffer size per side")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--spec-gb-s", type=float, default=None,
                    help="vendor spec, printed as a ratio if given")
    args = ap.parse_args()

    n = args.mib * 1024 * 1024 // 2   # fp16 elements
    a = torch.empty(n, dtype=torch.float16, device=args.device)
    b = torch.empty_like(a)

    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()

    best = 0.0
    for _ in range(args.iters):
        t0 = time.perf_counter()
        b.copy_(a)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        # Read once and write once, so the traffic is twice the buffer.
        best = max(best, 2 * a.numel() * a.element_size() / dt / 1e9)

    print(f"device                 {torch.cuda.get_device_name(0)}")
    print(f"buffer                 {args.mib} MiB fp16, {args.iters} iterations")
    print(f"measured bandwidth     {best:.1f} GB/s (read+write, best of "
          f"{args.iters})")
    if args.spec_gb_s:
        print(f"vendor spec            {args.spec_gb_s:.1f} GB/s")
        print(f"measured / spec        {best/args.spec_gb_s*100:.0f}%")


if __name__ == "__main__":
    main()
