# KVQuant Adversarial Technical Audit — Analysis Scripts

This directory contains four Python scripts that **verify**, through quantitative
analysis of the source code and toy simulation, the theoretical vulnerabilities
and bottlenecks identified in the KVQuant paper
([NeurIPS 2024, arXiv:2401.18079](https://arxiv.org/abs/2401.18079)).

These scripts are designed to support an adversarial technical audit for a
course project. Each script is **self-contained**, requires only the Python
standard library plus `math`, `re`, and `pathlib`, and can be run without a
GPU.

---

## Quick Start

```bash
# From the repository root
python analysis/cuda_kernel_complexity.py
python analysis/sparse_overhead_simulation.py
python analysis/dequantization_memory.py
python analysis/pipeline_integration_complexity.py
```

---

## Scripts and Vulnerabilities Addressed

### 1. `cuda_kernel_complexity.py` — Custom CUDA Kernel Complexity

**Claim under audit:** The paper describes its CUDA kernels as "efficient" and
"optimized" without disclosing their maintenance and extensibility cost.

**What this script does:**
- Counts all `__global__` kernel functions in `deployment/kvquant/quant_cuda_kernel.cu`
  (5 668 lines) and `benchmarking/kvquant/quant_cuda_kernel.cu` (1 254 lines).
- Classifies kernels along six independent axes of variation:
  bit-width (2/3/4), K vs V cache, sparse vs dense, RoPE vs no-RoPE,
  parallel vs sequential, and Orig vs Opt.
- Computes the theoretical maximum number of variants (combinatorial product)
  and the coverage ratio.
- Counts `atomicAdd` calls (48 in deployment) that serialize GPU threads.
- Identifies inline `cosf`/`sinf`/`powf` calls inside the decode loop that
  are recomputed per element rather than precomputed.

**Key finding:**
The deployment kernel contains **64 unique kernel functions** across a
**6-axis design space**, making it an incomplete cross-product. Adding a
single new feature (e.g., a new bit-width) requires implementing up to N
additional kernels by hand.

---

### 2. `sparse_overhead_simulation.py` — Dense-and-Sparse Quantization Overhead

**Claim under audit:** The paper presents the dense-and-sparse scheme as
practically efficient, without reporting the latency cost of building the
sparse KV cache.

**What this script does:**
- Models the **two-phase append pipeline** per token:
  1. GPU kernel: quantize + detect outliers
  2. Mandatory CPU-GPU sync: `outlier_count.to(torch::kCPU)` (line 745)
  3. GPU kernel: scatter outliers into COO arrays
  4. `torch::cat()`: grow sparse COO arrays (O(nnz_so_far) copy)
- Produces tables showing:
  - Memory footprint (fp16 baseline vs KVQuant) across context lengths.
  - Per-token append overhead (µs) as a function of context length.
  - Cumulative append overhead (seconds) for full-context prefill.

**Key finding:**
The mandatory CPU-GPU synchronization on every token append and the O(seqlen²)
total data copied by `torch::cat()` mean that building the KV cache for
**1M-token context takes tens of seconds of overhead** beyond the GPU compute,
a cost not reported in the paper.

---

### 3. `dequantization_memory.py` — Dequantization Bottleneck

**Claim under audit:** The paper claims near-lossless quantization with
efficient dequantization via its NUQ lookup table.

**What this script does:**
- Computes the **shared-memory allocation** for the LUT inside the fused
  NUQ+RoPE kernel (`deq2[17][BLOCKWIDTH]` = 8 704 B per block).
- Derives the maximum SM occupancy limited by shared-memory pressure.
- Compares the **arithmetic intensity** (FLOP/byte) of the NUQ+RoPE kernel
  vs a standard fp16 kernel, accounting for inline trig function cost.
- Counts the total number of transcendental function evaluations
  (cosf + sinf) per generated token at various context lengths.

**Key finding:**
The LUT (256 KB for LLaMA-7B at 4-bit) does **not fit in L1 cache** (32 KB).
SM occupancy is reduced to ~58% due to shared-memory pressure. At 1M-token
context, inline RoPE trig recomputation adds over **2.6 × 10¹¹**
transcendental evaluations per generated token — a compute cost that the paper
does not measure or report.

---

### 4. `pipeline_integration_complexity.py` — Integration Complexity

**Claim under audit:** The paper presents KVQuant as a practical, deployable
system. The README says "follow the README in each subfolder."

**What this script does:**
- Inventories all **20 user-facing hyperparameters** with their types, stages,
  and known interactions.
- Maps the **4-stage pipeline** dependency graph (gradients → quant →
  deployment → benchmarking), noting that each stage requires a separate
  conda environment and that Stage 1 embeds a 70,000-file fork of the
  Hugging Face Transformers library.
- Estimates the **KMeans calibration cost** in wall-clock time and memory for
  LLaMA-7B through LLaMA-65B.
- Documents **5 silent failure modes** — conditions that produce incorrect
  output without raising an exception.

**Key finding:**
Reproducing the paper's results requires compiling a 5 668-line CUDA extension,
running hundreds to thousands of CPU KMeans fitting calls, and correctly
configuring 20 interdependent hyperparameters across 3+ conda environments.
Several hyperparameter combinations are invalid but not validated.

---

## Mapping to Course Project Sections

| Analysis Script | Report Section | Rubric Criterion |
|---|---|---|
| `cuda_kernel_complexity.py` | §II Part A (System Reconstruction) | Technical Depth |
| `sparse_overhead_simulation.py` | §II Part B (Failure Conditions) | Critical Rigor |
| `dequantization_memory.py` | §II Part C (Scalability Analysis) | Scalability Vision |
| `pipeline_integration_complexity.py` | §III (Simulated Peer Review) | Hidden Costs / Practicality |

---

## Citations

These scripts analyze the open-source implementation accompanying:

```bibtex
@article{hooper2024kvquant,
  title={KVQuant: Towards 10 Million Context Length LLM Inference
         with KV Cache Quantization},
  author={Hooper, Coleman and Kim, Sehoon and Mohammadzadeh, Hiva
          and Mahoney, Michael W and Shao, Yakun Sophia
          and Keutzer, Kurt and Gholami, Amir},
  journal={arXiv preprint arXiv:2401.18079},
  year={2024}
}
```
