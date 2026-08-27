"""Plot the observation that motivated the dispatch experiment.

Three rented boxes, same script, same model, same workload, batch 1, eager
HuggingFace. Ranked by memory bandwidth the order is 3070 Ti, 3070, 4060 Ti.
Ranked by decode throughput it is exactly reversed. Decode at batch 1 is limited
by streaming the weights, so more bandwidth should mean more tokens per second,
and it did not.

Decode rates are medians read out of the committed baseline CSVs. Spec bandwidth
and two of the three CPU names are NOT in those files: the harness only gained a
cpu_model field partway through this work, so the 4060 Ti and 3070 Ti hosts come
from notes written when those boxes were rented. That provenance gap is stated in
the writeup too, and the mapping below is the single place it lives in code.

This figure shows a correlation across machines that differ in CPU, GPU, and
torch build simultaneously. It is the reason an experiment was run, not evidence
of cause on its own.

Usage:
    python scripts/plot_bandwidth_inversion.py \
        --plot results/bandwidth_inversion.png
"""

from __future__ import annotations

import argparse
import csv
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEIGHTS_GB = 2945.29052734375 * 1024 ** 2 / 1e9   # fp16 weights read per token

# card -> (csv, spec bandwidth GB/s, host CPU, CPU year)
# Bandwidth is the vendor figure. Only the RTX 3070 has a measured achievable
# number (402.6 GB/s); using spec for all three keeps the column comparable, and
# understates every MBU by roughly the 10% gap measured on that card.
CARDS = [
    ("RTX 4060 Ti", "results/rtx4060ti/baseline_hf.csv", 288, "i7-11700", 2021),
    ("RTX 3070", "results/rtx3070/baseline_hf.csv", 448, "Ryzen 7 3700X", 2019),
    ("RTX 3070 Ti", "results/rtx3070ti/baseline_hf.csv", 608, "Xeon E5-2680 v4", 2016),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", required=True)
    args = ap.parse_args()

    points = []
    for card, path, bw, cpu, year in CARDS:
        rows = list(csv.DictReader(open(path)))
        rates = [float(r["decode_tokens_per_sec"]) for r in rows]
        points.append({
            "card": card, "bw": bw, "cpu": cpu, "year": year,
            "decode": statistics.median(rates), "n": len(rates),
            "mbu": statistics.median(rates) * WEIGHTS_GB / bw * 100,
        })

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    xs = [p["bw"] for p in points]
    ys = [p["decode"] for p in points]

    ax.plot(xs, ys, "-", color="tab:red", lw=1.6, zorder=1)
    ax.scatter(xs, ys, s=90, color="tab:red", zorder=2)

    for p in points:
        ax.annotate(
            f"{p['card']}\n{p['cpu']} ({p['year']})\n"
            f"{p['decode']:.2f} tok/s, {p['mbu']:.1f}% MBU",
            xy=(p["bw"], p["decode"]),
            xytext=(0, 16 if p["card"] != "RTX 3070 Ti" else -52),
            textcoords="offset points", ha="center", fontsize=8.5,
        )

    ax.set_xlabel("GPU memory bandwidth (GB/s, vendor spec)")
    ax.set_ylabel("HuggingFace decode (tokens/s, batch 1)")
    ax.set_title(
        "More bandwidth, less throughput\n"
        "Qwen2.5-1.5B fp16, batch 1, eager transformers, same script on three boxes",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.set_xlim(240, 660)
    ax.set_ylim(0, max(ys) * 1.32)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)

    print(f"{'card':<14}{'BW':>6}{'decode':>10}{'MBU':>8}{'runs':>6}  host CPU")
    for p in points:
        print(f"{p['card']:<14}{p['bw']:>6}{p['decode']:>10.2f}"
              f"{p['mbu']:>7.1f}%{p['n']:>6}  {p['cpu']}")
    print(f"\nbandwidth ratio 3070 Ti / 4060 Ti = {608/288:.2f}x")
    print(f"decode ratio    3070 Ti / 4060 Ti = "
          f"{points[2]['decode']/points[0]['decode']:.2f}x")
    print(f"plot saved -> {args.plot}")


if __name__ == "__main__":
    main()
