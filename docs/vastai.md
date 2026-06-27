# Environment and plan changes (Vast.ai RTX 3070 Ti box)

Read alongside CLAUDE.md and phase1.md. This file records the actual compute we landed on and what it changes versus the original Colab-based plan. Where this file and CLAUDE.md disagree on hardware, this file wins.

## Three cards, on purpose

All single-GPU, same model and method, kept for a cross-hardware comparison:

- RTX 3060 12GB (collaborator's box): the first HF baseline, vLLM baseline, and OOM sweep. Results in `docs/baseline-hf-results.md`, `docs/baseline-vllm-results.md`, and the committed OOM plot. Labeled 3060, never relabel.
- RTX 4060 Ti 8GB (now deleted): a second set of runs. Results under `results/rtx4060ti/` if they were pushed before the box was destroyed.
- RTX 3070 Ti 8GB (current box): all new runs go under `results/rtx3070ti/`.

The 3070 Ti vs the 4060 Ti is the cleanest controlled pair: same 8GB, near-identical fp32 compute (~21-22 TFLOPS), but the 3070 Ti has ~2x the memory bandwidth (~608 GB/s GDDR6X vs 288 GB/s GDDR6). Decode is memory-bound, so expect decode ~2x faster on the 3070 Ti while prefill stays about the same. That contrast is itself a result worth plotting.

## What changed: compute and workflow

- We are no longer using Colab. Phase 1 runs on a rented Vast.ai box: one RTX 3070 Ti 8GB, on-demand, about $0.116/hr, located in South Korea (instance 42841519).
- Workflow is "Claude Code on the box," not local-first: edit, run on the GPU, read tracebacks, and commit in one place.
- Persistent working directory is `/workspace`. Storage is billed while stopped and a destroy wipes `/workspace` (the previous 4060 Ti box was destroyed, taking its local state). So push results to GitHub after every run. The repo is the durable artifact.

## Box specs (from the Vast.ai listing, confirm on the box)

- GPU: NVIDIA GeForce RTX 3070 Ti, 8GB (GDDR6X).
- Architecture: Ampere GA104, compute capability 8.6 (sm_86).
- Compute: ~21.7 fp32 TFLOPS. Memory bandwidth ~608 GB/s spec (256-bit, 19 Gbps GDDR6X); the listing shows 523.8 GB/s, verify on the box since decode is memory-bound and this sets the decode ceiling.
- Image: `vastai/pytorch_cuda-13.0.3-auto/jupyter`, Max CUDA 13.0. torch is preinstalled for CUDA 13.0; verify the exact torch version on the box.
- Host: Xeon E5-2680 v4 (Broadwell, 14 of 28 vCPU), 32GB RAM, 32GB disk, PCIE 3.0/8x. Older CPU than the 4060 Ti box's i7-11700, so HF eager decode may carry more host overhead here; watch the MBU.
- NOT preinstalled: transformers, vllm, accelerate. These are ours to add.
- Network: HuggingFace and GitHub both reachable, so weight downloads and git push both work.

## On startup, verify the GPU is idle before measuring

The marketplace listing can show the card busy while the auto-image initializes. Before any run, confirm `nvidia-smi` shows ~0% util and only a few MiB used, otherwise a stray process pollutes the measurement.

## Concrete steps (two environments, on purpose)

Environment A, the HF baseline:

```
export HF_HOME=/workspace/hf-cache
pip install -r requirements.txt
mkdir -p results/rtx3070ti
```

Environment B, the vLLM run, in an isolated venv so it cannot disturb A:

```
python -m venv /workspace/vllm-env
source /workspace/vllm-env/bin/activate
pip install uv
uv pip install vllm --torch-backend=auto    # auto matches the box's CUDA (13.0 here)
export HF_HOME=/workspace/hf-cache
```

Same model, prompt, max tokens, and batch 1 across both, so the comparison stays fair.

## One tooling note

Claude Code needs Node, which the image likely does not include. If `node --version` fails, install a current Node first, then `npm install -g @anthropic-ai/claude-code`. Set git identity (`git config --global user.name` / `user.email`) so commits from the box are attributed correctly.
</content>
