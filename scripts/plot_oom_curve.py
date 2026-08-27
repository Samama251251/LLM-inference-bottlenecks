"""Re-plot the KV-cache OOM curve from a committed CSV.

Standalone and torch-free on purpose: the CSV is the durable artifact, so the
plot must be regenerable on a laptop long after the rented box is gone. This is
the script to run when the figure needs to change, not oom_sweep.py, which would
require re-renting a GPU and re-running the whole sweep.

What this fixes versus the original inline plot in oom_sweep.py:

  1. The old figure drew peak RESERVED only. Reserved is what the caching
     allocator has grabbed from the driver, and it saturates once the allocator
     owns essentially the whole card, so the curve went flat well before the
     crash (around 43k tokens of 123k on the 3060). More than half the x-axis
     carried no information, and the measured-vs-predicted comparison died
     exactly where it got interesting.
  2. Peak ALLOCATED, which is what generation actually holds live, was recorded
     in the CSV but never drawn. It is the line that tracks the physics: linear
     in context length, all the way to the crash.

Drawing both is the point. Allocated says how memory grows; reserved says why
the process dies where it does, since it is reserved, not allocated, that has to
fit inside the card. On both measured cards the crash comes with roughly 1.3 GB
of allocator fragmentation between the two.

The lower panel plots decode throughput against context length, which the old
figure omitted entirely. It falls by about 10x across the sweep: attention has to
read the whole KV cache for every new token, so per-token cost grows with
context even though the weights read stays constant.

Usage:
    python scripts/plot_oom_curve.py \
        --csv results/rtx4060ti/oom_sweep.csv \
        --plot results/rtx4060ti/oom_curve.png \
        --card-gib 8
"""

from __future__ import annotations

import argparse
import csv
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIB = 1024 ** 2

# Qwen2.5-1.5B: 28 layers, 2 KV heads (GQA, 12 query heads share them), head_dim
# 128, fp16. 2 (K and V) x 28 x 2 x 128 x 2 bytes = 28,672 bytes per token.
DEFAULT_KV_BYTES_PER_TOKEN = 2 * 28 * 2 * 128 * 2

# Card capacities we have measured on, so the figure can label itself from the
# gpu_name column instead of needing the capacity passed in every time.
CARD_GIB = {
    "3060": 12,
    "4060 Ti": 8,
    "3070 Ti": 8,
}


def load_rows(path: str) -> list[dict]:
    """Read the sweep CSV, repairing the one known corruption in place.

    results/rtx3060/baseline_hf.csv was committed with its header glued to the first data
    row (a run killed before its newline landed). The same failure can hit a
    sweep CSV, so the reader tolerates it rather than dying on a stale file.
    """
    raw = open(path).read()
    for engine in ("hf", "vllm"):
        raw = raw.replace(f"timestamp{engine},", f"timestamp\n{engine},")
    return list(csv.DictReader(io.StringIO(raw)))


def infer_card_gib(gpu_name: str) -> int | None:
    for key, gib in CARD_GIB.items():
        if key in gpu_name:
            return gib
    return None


def fit_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and intercept, so the figure can state the measured
    bytes-per-token instead of leaving the reader to eyeball it."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return slope, (sy - slope * sx) / n


def render(
    csv_path: str,
    plot_path: str,
    card_gib: int | None = None,
    kv_bytes_per_token: int = DEFAULT_KV_BYTES_PER_TOKEN,
    annotate: bool = True,
) -> None:
    """Read a sweep CSV and write the figure. Importable so oom_sweep.py can
    call it directly after a run instead of keeping a second copy of the
    plotting code that can drift out of sync."""
    args = argparse.Namespace(
        csv=csv_path, plot=plot_path, card_gib=card_gib,
        kv_bytes_per_token=kv_bytes_per_token,
    )
    # annotate=False draws the same data without the box that explains the gap.
    # The writeup that introduces this figure builds to that explanation, so the
    # figure must not give it away before the reader gets there.
    rows = load_rows(args.csv)
    ok = [r for r in rows if r["oom"] == "False"]
    crashed = [r for r in rows if r["oom"] == "True"]
    if not ok:
        raise SystemExit("no non-OOM checkpoints in the CSV, nothing to plot")

    gpu = ok[0]["gpu_name"].strip()
    card_gib = args.card_gib or infer_card_gib(gpu)
    if card_gib is None:
        raise SystemExit(f"could not infer capacity for {gpu!r}; pass --card-gib")
    card_mib = card_gib * 1024

    # Context length is prompt + generated, not generated alone: the prompt sits
    # in the same KV cache.
    ctx = [int(r["prompt_tokens"]) + int(r["new_tokens"]) for r in ok]
    alloc = [float(r["peak_allocated_mib"]) for r in ok]
    reserved = [float(r["peak_reserved_mib"]) for r in ok]
    weights = float(ok[0]["weights_vram_mib"])
    predicted = [weights + c * args.kv_bytes_per_token / MIB for c in ctx]

    slope, _ = fit_slope([float(c) for c in ctx], alloc)
    measured_bpt = slope * MIB
    ratio = measured_bpt / args.kv_bytes_per_token

    oom_ctx = None
    if crashed:
        c = crashed[0]
        oom_ctx = int(c["prompt_tokens"]) + int(c["new_tokens"])

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    ax.plot(ctx, reserved, "-", color="tab:orange", lw=1.6,
            label="measured peak reserved (what must fit in the card)")
    ax.plot(ctx, alloc, "-o", color="tab:red", ms=2.5, lw=1.4,
            label=f"measured peak allocated ({measured_bpt/1024:.0f} KiB/token)")
    ax.plot(ctx, predicted, "--", color="tab:blue", lw=1.6,
            label=f"predicted: weights + analytical KV "
                  f"({args.kv_bytes_per_token/1024:.0f} KiB/token)")
    ax.axhline(card_mib, color="gray", ls=":", lw=1.2,
               label=f"{card_gib} GB card")
    if oom_ctx:
        ax.axvline(oom_ctx, color="black", ls="-.", lw=1.2,
                   label=f"CUDA OOM at ~{oom_ctx:,} tokens")

    ax.set_ylabel("GPU memory (MiB)")
    ax.set_title(
        f"The KV cache hits the wall, {ratio:.2f}x faster than the math predicts\n"
        f"Qwen2.5-1.5B fp16, {gpu}, HF transformers, grown by decode (1 token/step)",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, card_mib * 1.06)

    # State the headline gap on the figure rather than in a caption somewhere
    # else, so the plot cannot drift away from its own explanation.
    if annotate:
        ax.annotate(
            f"measured growth is {ratio:.2f}x the analytical KV line\n"
            f"({measured_bpt:,.0f} B/token vs {args.kv_bytes_per_token:,} B/token);\n"
            f"the cache itself is exactly 1.00x; the excess is a\n"
            f"per-step reallocation transient (see kv-cache-growth.md)",
            xy=(0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.9),
        )
    else:
        ax.annotate(
            f"measured growth is {ratio:.2f}x the analytical KV line\n"
            f"({measured_bpt:,.0f} B/token vs {args.kv_bytes_per_token:,} B/token)",
            xy=(0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.9),
        )

    rates = [float(r["decode_tokens_per_sec"]) for r in ok]
    pts = [(c, v) for c, v in zip(ctx, rates) if v > 0]
    if pts:
        ax2.plot([p[0] for p in pts], [p[1] for p in pts],
                 "-o", color="tab:green", ms=2.5, lw=1.4)
        first, last = pts[0][1], pts[-1][1]
        ax2.annotate(
            f"decode slows {first/last:.1f}x across the sweep\n"
            f"({first:.0f} to {last:.1f} tok/s): attention reads the\n"
            f"whole cache for every new token",
            xy=(0.97, 0.75), xycoords="axes fraction", ha="right", va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round", fc="honeydew", ec="gray", alpha=0.9),
        )
    if oom_ctx:
        ax2.axvline(oom_ctx, color="black", ls="-.", lw=1.2)
    ax2.set_xlabel("context length (tokens)")
    ax2.set_ylabel("decode (tok/s)")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    print(f"gpu                  {gpu} ({card_gib} GB)")
    print(f"checkpoints          {len(ok)}")
    print(f"context range        {ctx[0]:,} -> {ctx[-1]:,} tokens")
    print(f"measured allocated   {measured_bpt:,.0f} B/token")
    print(f"analytical KV        {args.kv_bytes_per_token:,} B/token")
    print(f"ratio                {ratio:.2f}x")
    if crashed:
        c = crashed[0]
        a, r = float(c["peak_allocated_mib"]), float(c["peak_reserved_mib"])
        print(f"OOM at               {oom_ctx:,} tokens")
        print(f"  allocated {a:,.0f} MiB / reserved {r:,.0f} MiB of {card_mib:,} "
              f"({r/card_mib*100:.1f}% reserved, {r-a:,.0f} MiB fragmentation)")
    print(f"plot saved -> {args.plot}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--plot", required=True)
    ap.add_argument("--card-gib", type=int, default=None,
                    help="card capacity; inferred from gpu_name when omitted")
    ap.add_argument("--kv-bytes-per-token", type=int,
                    default=DEFAULT_KV_BYTES_PER_TOKEN)
    ap.add_argument("--no-annotate", action="store_true",
                    help="omit the box explaining the gap, for use in writing "
                         "that builds to that explanation itself")
    a = ap.parse_args()
    render(a.csv, a.plot, a.card_gib, a.kv_bytes_per_token,
           annotate=not a.no_annotate)


if __name__ == "__main__":
    main()
