"""Plot the vLLM batching sweep: where throughput stops being free.

Reads the CSV written by bench_vllm_batch.py and draws the throughput and
latency curves against batch size. Standalone and torch-free, so the figure can
be regenerated on a laptop from committed data long after the box is gone.

This file previously existed as a byte-identical copy of bench_vllm_batch.py,
so the batching data was collected and never plotted. This is the actual
plotting code.

What the figure is meant to show, and what the committed data actually says:

  - Up to batch 16 throughput scales almost linearly (89% of ideal) while TPOT
    barely moves. Decode at batch 1 is memory-bound and the GPU is mostly idle
    waiting on weight reads, so extra sequences ride along in the same weight
    traffic nearly for free. This is the region that makes serving economical.
  - From 32 to 128 the free lunch ends. Throughput keeps climbing but scaling
    efficiency falls to 50%, and TPOT roughly doubles. The batch is now large
    enough that the GEMMs are compute-bound, so more work costs proportionally
    more time.
  - Past the KV pool's capacity the scheduler cannot run every sequence in one
    wave. It runs a subset and defers the rest, `queued` flips true, and
    aggregate throughput does not merely flatten, it drops: the realistic run
    peaks at batch 128 and falls back at 192 and 256.

The KV wall is visible as queuing, not as rising memory. vLLM reserves its KV
pool at startup, so device VRAM is roughly flat across the whole sweep and you
cannot watch memory fill up the way the HF OOM sweep does. `kv_capacity_seqs`
from the engine's own block accounting is what predicts where the wall lands,
and it does so accurately in both configurations.

Usage:
    python scripts/plot_batch_sweep.py \
        --csv results/rtx4060ti/vllm_batch_sweep.csv \
        --plot results/rtx4060ti/batch_curve.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One colour per pool configuration, so the two runs stay visually distinct
# across both panels.
STYLE = {
    "realistic": ("tab:blue", "o", "realistic pool (gpu_mem_util 0.9)"),
    "constrained": ("tab:red", "s", "constrained pool (gpu_mem_util 0.65)"),
}


def load(path: str) -> dict[str, list[dict]]:
    """Group rows by pool_config, averaging the repeats at each batch size."""
    rows = list(csv.DictReader(open(path)))
    by_config: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_config[r["pool_config"]][int(r["batch_size"])].append(r)

    out: dict[str, list[dict]] = {}
    for cfg, by_batch in by_config.items():
        points = []
        for batch in sorted(by_batch):
            reps = by_batch[batch]
            n = len(reps)
            points.append({
                "batch": batch,
                "n_repeats": n,
                "throughput": sum(float(x["throughput_out_tok_s"]) for x in reps) / n,
                "decode_throughput": sum(float(x["decode_throughput_tok_s"]) for x in reps) / n,
                "tpot_ms": sum(float(x["tpot_ms"]) for x in reps) / n,
                "ttft_s": sum(float(x["ttft_s"]) for x in reps) / n,
                # A batch counts as queued if any repeat queued: the wall is a
                # capacity fact, not an averageable quantity.
                "queued": any(x["queued"] == "True" for x in reps),
                "kv_capacity_seqs": int(reps[0]["kv_capacity_seqs"]),
                "device_used_mib": sum(float(x["device_used_mib"]) for x in reps) / n,
            })
        out[cfg] = points
    return out


def render(csv_path: str, plot_path: str) -> None:
    data = load(csv_path)
    if not data:
        raise SystemExit(f"no rows in {csv_path}")

    meta = next(iter(csv.DictReader(open(csv_path))))
    gpu = meta["gpu_name"].strip()
    model = meta["model"]
    vllm_version = meta.get("vllm_version", "?")

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1]},
    )

    labelled: set[str] = set()
    for cfg, points in data.items():
        colour, marker, label = STYLE.get(cfg, ("tab:gray", "^", cfg))
        batches = [p["batch"] for p in points]
        thru = [p["decode_throughput"] for p in points]

        ax.plot(batches, thru, "-", marker=marker, color=colour, ms=5, lw=1.6,
                label=label)

        # Mark the points where the scheduler had to defer work.
        qx = [p["batch"] for p in points if p["queued"]]
        qy = [p["decode_throughput"] for p in points if p["queued"]]
        if qx:
            # Label once: the marker means the same thing in both runs, and one
            # legend entry per config would just repeat itself.
            ax.plot(qx, qy, "x", color="black", ms=11, mew=2.2, zorder=5,
                    label="requests queued (past KV capacity)"
                    if "queued" not in labelled else None)
            labelled.add("queued")

        # The engine's own KV block accounting predicts the wall. Draw it.
        cap = points[0]["kv_capacity_seqs"]
        ax.axvline(cap, color=colour, ls=":", lw=1.4, alpha=0.75)
        ax.annotate(f"KV pool holds\n{cap} sequences", xy=(cap, max(thru) * 0.42),
                    fontsize=8, color=colour, ha="right", rotation=90,
                    va="center")

        ax2.plot(batches, [p["tpot_ms"] for p in points], "-", marker=marker,
                 color=colour, ms=5, lw=1.6, label=label)
        if qx:
            ax2.plot(qx, [p["tpot_ms"] for p in points if p["queued"]], "x",
                     color="black", ms=11, mew=2.2, zorder=5)

    # Ideal linear scaling from the batch-1 point, so the reader can see the
    # free-lunch region peel away from it rather than taking it on faith.
    ref = data.get("realistic") or next(iter(data.values()))
    base = ref[0]["decode_throughput"]
    xs = [p["batch"] for p in ref]
    ax.plot(xs, [base * x for x in xs], "--", color="gray", lw=1.2,
            label="ideal linear scaling")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylabel("decode throughput (tok/s, all sequences)")
    ax.set_title(
        f"Batching is free until the KV pool runs out\n"
        f"{model} fp16, {gpu}, vLLM {vllm_version}",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3, which="both")

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("batch size (concurrent sequences in one generate call)")
    ax2.set_ylabel("TPOT (ms per output token)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(x) for x in xs])
    ax2.annotate(
        "latency is flat while throughput is free,\n"
        "climbs once decode goes compute-bound,\n"
        "then jumps when requests start queuing",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=130)

    for cfg, points in data.items():
        cap = points[0]["kv_capacity_seqs"]
        peak = max(points, key=lambda p: p["decode_throughput"])
        base_t = points[0]["decode_throughput"]
        first_q = next((p["batch"] for p in points if p["queued"]), None)
        print(f"\n{cfg}: KV pool holds {cap} sequences, "
              f"{points[0]['n_repeats']} repeats per point")
        print(f"  peak decode throughput {peak['decode_throughput']:,.0f} tok/s "
              f"at batch {peak['batch']} "
              f"({peak['decode_throughput']/base_t:.1f}x batch-1, "
              f"{peak['decode_throughput']/base_t/peak['batch']*100:.0f}% of ideal)")
        if first_q:
            after = [p for p in points if p["batch"] > first_q]
            print(f"  first queued at batch {first_q} (predicted {cap})")
            if after:
                worst = min(after, key=lambda p: p["decode_throughput"])
                print(f"  past the wall throughput falls to "
                      f"{worst['decode_throughput']:,.0f} tok/s at batch {worst['batch']}, "
                      f"TPOT {worst['tpot_ms']:.0f} ms")
        used = [p["device_used_mib"] for p in points]
        print(f"  device VRAM {min(used):,.0f} to {max(used):,.0f} MiB "
              f"(reserved pool, so the wall is queuing and not bytes)")
    print(f"\nplot saved -> {plot_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--plot", required=True)
    a = ap.parse_args()
    render(a.csv, a.plot)


if __name__ == "__main__":
    main()
