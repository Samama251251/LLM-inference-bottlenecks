# Environment and plan changes (Vast.ai RTX 4060 Ti box)

Read alongside CLAUDE.md and phase1.md. This file records the actual compute we landed on and what it changes versus the original Colab-based plan. Where this file and CLAUDE.md disagree on hardware, this file wins.

## Two boxes, on purpose

- Current box (ours): one RTX 4060 Ti 8GB. All new Phase 1 measurements run here.
- Earlier box (collaborator): one RTX 3060 12GB. A collaborator ran the first HF baseline, vLLM baseline, and OOM sweep there. We keep those results (`docs/baseline-hf-results.md`, `docs/baseline-vllm-results.md`, and the committed OOM plot) as a cross-hardware comparison baseline: Ampere 12GB vs Ada 8GB, same model and method. Those numbers stay labeled as 3060. Do not relabel them.

## What changed: compute and workflow

- We are no longer using Colab. Phase 1 runs on a rented Vast.ai box: one RTX 4060 Ti 8GB, on-demand, about $0.095/hr, located in Canada (instance 42824740).
- Workflow is "Claude Code on the box," not local-first. Claude Code runs over SSH directly on the rented machine, so it edits, runs on the GPU, reads tracebacks, and commits all in one place. No git push/pull loop between a laptop and the runner.
- Persistent working directory is `/workspace`. Storage is billed while the instance is stopped, so the repo and the weights cache live in `/workspace`, not in `/root` or `/tmp`. Disk is only 32GB, so be deliberate about the weights cache. Destroy the instance when done for the week.

## Box specs (from the Vast.ai listing, confirm on the box)

- GPU: NVIDIA GeForce RTX 4060 Ti, 8GB (8192 MiB).
- Architecture: Ada Lovelace, compute capability 8.9 (sm_89). Adds fp8 over Ampere.
- Compute: ~22 fp32 TFLOPS. Memory bandwidth ~288 GB/s spec (the listing shows 233.1 GB/s; verify on the box, since decode is memory-bound and this number sets the decode ceiling).
- Image: `vastai/pytorch_cuda-12.4.1-auto/jupyter`, Max CUDA 12.4. torch is preinstalled for CUDA 12.4; verify the exact torch version on the box (it is NOT the old NGC torch 2.12+cu130).
- Host: 11th Gen Intel Core i7-11700 (16 vCPU), 64GB RAM, 32GB disk, Z590 board, PCIE 4.0/8x.
- NOT preinstalled: transformers, vllm, accelerate. These are ours to add.
- Network: HuggingFace and GitHub both reachable, so weight downloads and git push both work.

## What this changes in phase1.md

1. The vLLM attention-backend worry is resolved in our favor. phase1.md warned that a Turing T4 (sm_75) might force vLLM into a fallback backend. On Ada sm_89 that does not apply: FlashAttention and vLLM's good kernels support this card. Delete that worry.

2. The CUDA-stack friction is lower than on the old 3060 box. That box had torch 2.12+cu130, newer than what vLLM bundled, so a naive install would clobber it. This box is CUDA 12.4, which is vLLM's mainstream wheel target, so a fresh-env cu124 install is the well-trodden path.

   Rule still holds: install vLLM in a separate fresh virtual environment and let the wheel bring its own torch. Do not try to make vLLM reuse the system torch (that path means building from source).

3. fp16 still stands. This card supports bf16 and fp8, unlike the T4 and V100. We still target fp16, both for comparability with the later NUST HPC V100/T4 runs and the 3060 baseline, and because it does not change the KV-cache OOM math (fp16 and bf16 are both 2 bytes per element).

4. OOM math on 8GB. A 1.5B model in fp16 is about 3GB of weights, leaving roughly 5GB for the KV cache to grow into, versus ~9GB on the 3060 baseline. The OOM cliff lands notably sooner, which makes the sweep faster and cheaper and gives a clean two-card comparison of where the wall falls.

## Concrete next steps (two environments, on purpose)

Environment A, the HF baseline. Use the existing preinstalled-torch environment. Only add:

```
pip install transformers accelerate
export HF_HOME=/workspace/hf-cache   # cache weights on persistent storage
```

Build and validate the baseline harness here (Task 1 in phase1.md). This path has no version drama, so it should be a clean first win.

Environment B, the vLLM run. Create a fresh, isolated environment so it cannot disturb Environment A:

```
python -m venv /workspace/vllm-env
source /workspace/vllm-env/bin/activate
# install vLLM and let it pull its own matching torch for CUDA 12.4
# (uv is the recommended installer; uv pip install vllm --torch-backend=cu124)
pip install vllm
```

Run the vLLM benchmark (Task 2) from Environment B. Same model, same prompt, same max tokens, batch size 1, so the comparison against Environment A stays fair.

## One tooling note

Claude Code needs Node, which the image likely does not include. If `node --version` fails, install a current Node first, then `npm install -g @anthropic-ai/claude-code`. Set git identity (`git config --global user.name` / `user.email`) so commits from the box are attributed correctly.
</content>
</invoke>
