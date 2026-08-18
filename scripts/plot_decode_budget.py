"""Split the per-token decode time into weight read, other GPU work, and idle.

Sources, all committed artifacts:
  results/rtx3070/dispatch_proof.csv   wall time per token, both arms, and the
                                       weight-read floor at the measured bandwidth
  results/rtx3070/decode_profile.json  summed device-side kernel time per token
  results/rtx3070/baseline_vllm.csv    vLLM wall time per token

The GPU-busy figure was profiled on the eager path only. The compiled bar reuses
it, which is an assumption rather than a measurement, though the compiled wall
time landing within 0.1 ms of it is consistent with the kernels being the same
work replayed. vLLM was never profiled, so its bar shows total wall time only.

Usage:
    python scripts/plot_decode_budget.py --plot results/rtx3070/decode_budget.png
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispatch", default="results/rtx3070/dispatch_proof.csv")
    ap.add_argument("--profile", default="results/rtx3070/decode_profile.json")
    ap.add_argument("--vllm", default="results/rtx3070/baseline_vllm.csv")
    ap.add_argument("--plot", required=True)
    a = ap.parse_args()

    dp = list(csv.DictReader(open(a.dispatch)))
    prof = json.load(open(a.profile))
    vl = list(csv.DictReader(open(a.vllm)))

    floor = float(dp[0]["weight_read_ms"])
    busy = prof["gpu_busy_ms_per_token"]
    other_gpu = busy - floor

    def wall(mode):
        return statistics.median(
            float(r["ms_per_token"]) for r in dp if r["mode"] == mode)

    vllm_wall = 1000 / statistics.median(
        float(r["decode_tokens_per_sec"]) for r in vl)

    labels = ["eager\n(StaticCache)", "torch.compile\n(CUDA graphs)", "vLLM 0.24.0"]
    walls = [wall("eager"), wall("compiled"), vllm_wall]
    reads = [floor, floor, floor]
    others = [other_gpu, other_gpu, None]
    idles = [walls[0] - busy, walls[1] - busy, None]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    x = range(3)

    ax.bar(x, reads, 0.55, color="tab:blue", label=f"weight read ({floor:.2f} ms)")
    ax.bar(x[:2], others[:2], 0.55, bottom=reads[:2], color="tab:cyan",
           label=f"other GPU work ({other_gpu:.2f} ms, profiled on eager)")
    ax.bar(x[:2], idles[:2], 0.55, bottom=[busy, busy], color="tab:red",
           label="GPU idle, waiting on the host")
    # vLLM was not profiled, so its bar is drawn as a single unsplit total.
    ax.bar([2], [vllm_wall - floor], 0.55, bottom=[floor], color="lightgray",
           hatch="//", edgecolor="gray",
           label="vLLM: not profiled, split unknown")

    for i, w in enumerate(walls):
        ax.text(i, w + 0.35, f"{w:.2f} ms", ha="center", fontsize=10)

    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("time per decoded token (ms)")
    ax.set_title("Where a decoded token's time goes\n"
                 "Qwen2.5-1.5B fp16, RTX 3070, batch 1, 512-token prompt",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(walls) * 1.22)

    fig.tight_layout()
    fig.savefig(a.plot, dpi=130)
    for lbl, w in zip(labels, walls):
        print(f"{lbl.replace(chr(10),' '):<28}{w:>7.2f} ms/token")
    print(f"{'weight read floor':<28}{floor:>7.2f} ms")
    print(f"{'GPU busy (eager, profiled)':<28}{busy:>7.2f} ms")
    print(f"plot saved -> {a.plot}")


if __name__ == "__main__":
    main()
