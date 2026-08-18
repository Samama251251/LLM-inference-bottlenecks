"""Plot what the OOM curve was actually showing.

The OOM sweeps plotted peak allocated memory against context length and it rose
2.21x faster than the analytical KV-cache formula predicts, on two independent
cards, with no explanation. kv_growth_probe.py decomposed it. This draws the
decomposition.

The finding, in one figure: the KV cache is exactly the size the formula says.
Live allocated memory sits on the analytical line at 1.00x for every checkpoint
out to 20k tokens. What the OOM curve was tracking is peak, and peak carries a
per-step transient from cache reallocation, because the old cache is still live
while the new one is being built.

How big that transient is depends on which cache path transformers takes. The
legacy tuple-of-tuples API carries 2.22x the cache in transients; a DynamicCache
object carries 1.22x, exactly one full copy less. So the choice of cache
implementation, which changes no arithmetic and no output, decides how much
context fits on the card.

Usage:
    python scripts/plot_kv_growth.py \
        --legacy results/rtx3070/kv_growth_legacy.csv \
        --dynamic results/rtx3070/kv_growth_dynamic.csv \
        --plot results/rtx3070/kv_growth.png
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIB = 1024 ** 2
KV_PER_TOKEN = 2 * 28 * 2 * 128 * 2   # Qwen2.5-1.5B, 28,672 B/token


def load(path):
    return list(csv.DictReader(open(path)))


def slope(rows, key):
    xs = [int(r["seq_len"]) for r in rows]
    ys = [float(r[key]) for r in rows]
    n = len(xs); sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    return (n * sxy - sx * sy) / (n * sxx - sx * sx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", required=True)
    ap.add_argument("--dynamic", required=True)
    ap.add_argument("--plot", required=True)
    ap.add_argument("--card-gib", type=float, default=8)
    args = ap.parse_args()

    leg, dyn = load(args.legacy), load(args.dynamic)
    seq = [int(r["seq_len"]) for r in leg]
    weights = float(leg[0]["weights_b"]) / MIB

    analytical = [(weights + s * KV_PER_TOKEN / MIB) for s in seq]
    live = [float(r["total_allocated_b"]) / MIB for r in leg]
    peak_leg = [float(r["peak_allocated_b"]) / MIB for r in leg]
    peak_dyn = [float(r["peak_allocated_b"]) / MIB for r in dyn]

    fig, ax = plt.subplots(figsize=(9.5, 6))

    ax.plot(seq, peak_leg, "-o", color="tab:red", ms=3.5, lw=1.7,
            label=f"peak, legacy tuple cache  "
                  f"({slope(leg,'peak_allocated_b')/KV_PER_TOKEN:.2f}x analytical)")
    ax.plot(seq, peak_dyn, "-s", color="tab:orange", ms=3.5, lw=1.7,
            label=f"peak, DynamicCache  "
                  f"({slope(dyn,'peak_allocated_b')/KV_PER_TOKEN:.2f}x analytical)")
    ax.plot(seq, live, "-", color="tab:green", lw=2.4,
            label=f"live allocated  "
                  f"({slope(leg,'total_allocated_b')/KV_PER_TOKEN:.2f}x analytical)")
    ax.plot(seq, analytical, "--", color="tab:blue", lw=1.8,
            label="analytical: weights + 2 x L x kv_heads x head_dim x seq x 2B")

    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("GPU memory (MiB)")
    ax.set_title(
        "The KV cache is exactly the predicted size.\n"
        "What the OOM curve measured was the reallocation transient on top of it.",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax.annotate(
        "live memory sits ON the analytical line:\n"
        "the formula was never wrong",
        xy=(seq[-2], live[-2]), xytext=(seq[-1] * 0.44, live[-1] * 0.985),
        fontsize=9, color="tab:green",
        arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.3),
    )
    mid = len(seq) // 2
    ax.annotate(
        "DynamicCache removes exactly one full\n"
        "copy of the cache: 1.45x more context\n"
        "fits on the same card",
        xy=(seq[mid], peak_dyn[mid]),
        xytext=(seq[mid] * 1.06, peak_leg[mid] * 0.955),
        fontsize=9, color="tab:orange",
        arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.3),
    )
    # The vertical gap between the two peak curves is one cache copy; label it.
    ax.annotate(
        "", xy=(seq[-3], peak_leg[-3]), xytext=(seq[-3], peak_dyn[-3]),
        arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2),
    )
    ax.text(seq[-3] * 1.01, (peak_leg[-3] + peak_dyn[-3]) / 2,
            "1 cache copy", fontsize=8.5, color="gray", va="center")

    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)

    print(f"{'series':<26}{'B/token':>12}{'vs analytical':>15}")
    for lbl, rows, key in (
        ("analytical KV", leg, "predicted_kv_b"),
        ("live cache tensors", leg, "actual_kv_b"),
        ("live total allocated", leg, "total_allocated_b"),
        ("peak, legacy tuple", leg, "peak_allocated_b"),
        ("peak, DynamicCache", dyn, "peak_allocated_b"),
    ):
        s = slope(rows, key)
        print(f"{lbl:<26}{s:>12,.0f}{s/KV_PER_TOKEN:>14.2f}x")

    pl, pd = slope(leg, "peak_allocated_b"), slope(dyn, "peak_allocated_b")
    avail = args.card_gib * 1024 * MIB - float(leg[0]["weights_b"])
    print(f"\nusable context before the card is full ({args.card_gib:.0f} GiB, "
          f"weights {weights:.0f} MiB):")
    print(f"  legacy tuple   {avail/pl:>9,.0f} tokens")
    print(f"  DynamicCache   {avail/pd:>9,.0f} tokens   ({pl/pd:.2f}x more context)")
    print(f"\nplot saved -> {args.plot}")


if __name__ == "__main__":
    main()
