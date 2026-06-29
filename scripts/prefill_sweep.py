"""Prefill efficiency vs prompt length: watch the GEMMs fill the tensor cores.

The companion to the OOM sweep, on the other phase. Decode is memory-bound and
the OOM sweep walks the bandwidth/memory wall; prefill is compute-bound and this
sweep walks the compute wall. The question it answers: at batch 1, a 512-token
prefill runs at only ~38% of the tensor-core peak because the GEMM is small and
leaves the tensor cores half-empty. If we make the prompt longer, does the GEMM
get big enough to saturate them?

The prefill matmul is [M, K] x [K, N] with M = prompt tokens. cuBLAS tiles the
output into blocks and hands one block per SM; small M means partial tiles, low
occupancy, and launch overhead amortized over little work. Larger M packs full
tiles and keeps every SM busy, so effective TFLOPS climbs toward the roof and
then flattens. This script measures that curve.

Two honest caveats it also surfaces:
  1. Higher efficiency is not faster prefill. Total FLOPs grow with tokens, so
     TTFT (the bottom panel) goes UP even as efficiency improves. You get better
     value per FLOP, not a shorter wait.
  2. The 2 x params x tokens estimate counts only the linear layers, which grow
     linearly. Attention grows as seq^2 and is ~3% at 512 but tens of percent
     past a few thousand tokens, so this script counts it explicitly and the
     effective-TFLOPS number is over (linear + attention) FLOPs, not linear
     alone.

Usage (Environment A, the HF venv with the box's preinstalled torch):
    python scripts/prefill_sweep.py \
        --model Qwen/Qwen2.5-1.5B \
        --sizes 128,256,512,1024,2048,4096,8192 \
        --peak-tflops 87 \
        --csv results/rtx3070ti/prefill_sweep.csv \
        --plot results/rtx3070ti/prefill_curve.png
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_common import (
    BenchResult,
    CudaTimer,
    VramSnapshot,
    bytes_to_mib,
    print_env,
    reset_peak_vram,
    write_result,
)


def build_prompt_ids(tokenizer, target_tokens: int, device: str) -> torch.Tensor:
    """A length-exact prompt. Same seed-and-tile scheme as the other scripts."""
    seed = "The quick brown fox jumps over the lazy dog. "
    ids = tokenizer(seed, return_tensors="pt").input_ids[0]
    reps = (target_tokens // ids.numel()) + 1
    ids = ids.repeat(reps)[:target_tokens]
    return ids.unsqueeze(0).to(device)


def prefill_flops(param_count: int, cfg, seq_len: int) -> tuple[float, float]:
    """(linear_flops, attention_flops) for a single prefill pass of seq_len tokens.

    Linear layers: 2 x params x tokens, the standard forward-FLOP estimate (every
    weight participates in one multiply-add per token).

    Attention: two batched matmuls per layer, QK^T and softmax@V, each
    2 x num_q_heads x head_dim x seq^2 FLOP, so 4 x layers x q_heads x head_dim x
    seq^2 total. This is the seq^2 term the linear estimate ignores. We count it
    so the effective-TFLOPS number reflects the real work, not an undercount that
    would make long contexts look artificially efficient.
    """
    linear = 2.0 * param_count * seq_len

    num_layers = cfg.num_hidden_layers
    num_q_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    attention = 4.0 * num_layers * num_q_heads * head_dim * (seq_len ** 2)

    return linear, attention


@torch.inference_mode()
def time_prefill(model, input_ids, dev_idx) -> float:
    """One parallel forward over the whole prompt, timed. Returns seconds."""
    with CudaTimer(dev_idx) as t:
        out = model(input_ids=input_ids, use_cache=True)
        # force the logits to exist before the timer's synchronize, so we time the
        # full prefill and not a half-launched graph.
        _ = out.logits[:, -1, :].argmax(dim=-1)
    return t.seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--sizes",
        default="128,256,512,1024,2048,4096,8192",
        help="Comma-separated prompt lengths to sweep.",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=87.0,
        help="fp16 tensor-core peak for the % -of-peak line. ~87 for the RTX 3070 "
        "Ti (FP32 accumulate, dense), ~51 for the 3060. Set to your card.",
    )
    parser.add_argument(
        "--attn-impl",
        default="sdpa",
        help="Attention kernel. sdpa (default) uses the memory-efficient/flash "
        "path so long prompts do not materialize an seq x seq score matrix and "
        "OOM the prefill transient. eager would match the old 512-token baseline "
        "exactly but cannot reach 8k on 8 GB.",
    )
    parser.add_argument("--csv", default="results/prefill_sweep.csv")
    parser.add_argument("--plot", default="results/prefill_curve.png")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. This sweep measures GPU prefill; run it on the box.")

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    print_env()
    device = args.device
    dev_idx = torch.device(device).index or 0

    print(f"\nloading {args.model} in fp16 (attn={args.attn_impl}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation=args.attn_impl
    ).to(device)
    model.eval()

    torch.cuda.synchronize(dev_idx)
    weights_vram = bytes_to_mib(torch.cuda.memory_allocated(dev_idx))
    param_count = sum(p.numel() for p in model.parameters())
    print(f"weights resident: {weights_vram:.0f} MiB, params: {param_count / 1e9:.3f} B")
    print(f"compute floor uses fp16 tensor peak = {args.peak_tflops:.0f} TFLOPS\n")

    rows: list[dict] = []

    print(
        f"{'tokens':>7} | {'prefill ms':>10} | {'tok/s':>8} | "
        f"{'eff TFLOPS':>10} | {'% peak':>7} | {'attn %':>6}"
    )
    print("-" * 64)

    for n in sizes:
        input_ids = build_prompt_ids(tokenizer, n, device)
        actual = input_ids.shape[1]

        try:
            # per-size warmup (discarded): autotunes the kernel for THIS shape so
            # the measured time is steady-state, not first-call autotuning.
            _ = time_prefill(model, input_ids, dev_idx)

            reset_peak_vram(dev_idx)
            seconds = time_prefill(model, input_ids, dev_idx)
            vram = VramSnapshot.capture(dev_idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{actual:>7} |  OOM during prefill (transient too big for the card)")
            write_result(
                BenchResult(
                    engine="hf", model=args.model, dtype="float16", batch_size=1,
                    prompt_tokens=actual, new_tokens=0,
                    prefill_seconds=0.0, decode_seconds=0.0,
                    decode_tokens_per_sec=0.0, prefill_tokens_per_sec=0.0,
                    weights_vram_mib=weights_vram,
                    peak_allocated_mib=0.0, peak_reserved_mib=0.0,
                    oom=True, note="OOM during prefill",
                ),
                args.csv,
            )
            break

        linear_flops, attn_flops = prefill_flops(param_count, model.config, actual)
        total_flops = linear_flops + attn_flops
        eff_tflops = total_flops / seconds / 1e12
        pct_peak = 100.0 * eff_tflops / args.peak_tflops
        attn_pct = 100.0 * attn_flops / total_flops
        prefill_tps = actual / seconds

        note = (
            f"eff_tflops={eff_tflops:.1f} over linear+attn FLOPs; "
            f"pct_peak={pct_peak:.0f}% of {args.peak_tflops:.0f}; "
            f"attn_frac={attn_pct:.0f}%; attn={args.attn_impl}"
        )
        write_result(
            BenchResult(
                engine="hf", model=args.model, dtype="float16", batch_size=1,
                prompt_tokens=actual, new_tokens=0,
                prefill_seconds=seconds, decode_seconds=0.0,
                decode_tokens_per_sec=0.0, prefill_tokens_per_sec=prefill_tps,
                weights_vram_mib=weights_vram,
                peak_allocated_mib=vram.peak_allocated_mib,
                peak_reserved_mib=vram.peak_reserved_mib,
                oom=False, note=note,
            ),
            args.csv,
        )
        rows.append(
            {
                "tokens": actual,
                "prefill_ms": seconds * 1000.0,
                "tok_s": prefill_tps,
                "eff_tflops": eff_tflops,
                "pct_peak": pct_peak,
                "attn_pct": attn_pct,
            }
        )
        print(
            f"{actual:>7} | {seconds * 1000:>10.1f} | {prefill_tps:>8.0f} | "
            f"{eff_tflops:>10.1f} | {pct_peak:>6.0f}% | {attn_pct:>5.0f}%"
        )

        # free the KV cache and activations before the next, bigger size so the
        # transient does not stack across sizes.
        del input_ids
        torch.cuda.empty_cache()

    _maybe_plot(rows, args)
    print(f"\nrows appended -> {args.csv}")


def _maybe_plot(rows, args) -> None:
    """Two stacked panels: effective TFLOPS climbing toward the roof (the point),
    and prefill latency growing anyway (the caveat). CSV is the real artifact and
    is always written above, so a box without matplotlib still produces the data.
    """
    if not rows:
        print("no rows recorded; skipping plot")
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; CSV written, plot skipped")
        return

    tokens = [r["tokens"] for r in rows]
    eff = [r["eff_tflops"] for r in rows]
    ms = [r["prefill_ms"] for r in rows]

    dev_idx = torch.device(args.device).index or 0
    props = torch.cuda.get_device_properties(dev_idx)
    card_name = props.name

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # top: efficiency vs prompt length, with the tensor-core roof
    ax_top.plot(tokens, eff, "-o", color="tab:green", markersize=4,
                label="measured effective TFLOPS")
    ax_top.axhline(args.peak_tflops, color="gray", ls=":", lw=1,
                   label=f"fp16 tensor peak ~{args.peak_tflops:.0f} TFLOPS")
    for r in rows:
        ax_top.annotate(f"{r['pct_peak']:.0f}%",
                        (r["tokens"], r["eff_tflops"]),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7, color="tab:green")
    ax_top.set_ylabel("effective TFLOPS (linear + attention)")
    ax_top.set_ylim(0, args.peak_tflops * 1.1)
    ax_top.set_title(
        f"Prefill fills the tensor cores as the prompt grows\n"
        f"Qwen2.5-1.5B fp16, {card_name}, batch 1, HF transformers"
    )
    ax_top.legend(loc="lower right", fontsize=8)
    ax_top.grid(True, alpha=0.3)

    # bottom: the caveat, latency grows even as efficiency improves
    ax_bot.plot(tokens, ms, "-o", color="tab:red", markersize=4,
                label="prefill latency (TTFT)")
    ax_bot.set_xlabel("prompt length (tokens)")
    ax_bot.set_ylabel("prefill latency (ms)")
    ax_bot.set_xscale("log", base=2)
    ax_bot.set_xticks(tokens)
    ax_bot.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_bot.legend(loc="upper left", fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    print(f"plot saved -> {args.plot}")


if __name__ == "__main__":
    main()
