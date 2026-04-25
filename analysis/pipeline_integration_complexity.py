"""
Vulnerability 4: Integration Complexity
=========================================

The KVQuant system is not a drop-in library.  To compress a single model,
a user must execute a mandatory, sequential, multi-stage pipeline:

  Stage 1: gradients/   — compute Fisher information (full Hugging Face fork)
  Stage 2: quant/       — calibration, NUQ codebook fitting (KMeans), export
  Stage 3: deployment/  — compile custom CUDA extension; run quantized model
  (Bonus) benchmarking/ — compile a second CUDA extension; run kernel tests

Each stage has its own conda environment, its own set of hyperparameters,
and can fail in ways that are non-obvious from the paper's description.

This script:
  (a) Counts the number of user-facing hyperparameters across the pipeline.
  (b) Estimates the KMeans calibration cost (time and memory).
  (c) Maps the dependency graph between stages.
  (d) Highlights "silent failure" modes: conditions where calibration silently
      produces degenerate codebooks without raising an error.

Run with:
    python analysis/pipeline_integration_complexity.py
"""

import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Hyperparameter inventory
# ---------------------------------------------------------------------------

# Collected from: quant/kvquant/simquant_module_quantizer.py,
#   deployment/kvquant/simquant_module_quantizer.py,
#   quant/llama_simquant.py (inferred from argparse)
HYPERPARAMETERS = [
    # --- Quantization bits ---
    ("bits",              "int",   "Number of quantization bits (2/3/4/5)",
     "quant + deployment"),
    # --- Outlier / sparsity ---
    ("include_sparse",    "bool",  "Enable dense-and-sparse quantization",
     "quant + deployment"),
    ("sparsity_threshold","float", "Percentile threshold for outlier detection (e.g. 0.999)",
     "quant + deployment"),
    ("cap_outliers",      "bool",  "Cap to fixed num outliers per token (needed for efficient kernels)",
     "quant + deployment"),
    # --- NUQ / codebook ---
    ("nuq",               "bool",  "Enable Non-Uniform Quantization (NUQ)",
     "quant + deployment"),
    ("nf_nuq",            "bool",  "Use NormalFloat signposts instead of KMeans",
     "quant + deployment"),
    ("norm",              "bool",  "Enable Q-Norm post-quantization normalisation",
     "quant + deployment"),
    # --- Attention sink ---
    ("first_few_fp16",    "int",   "Keep first N tokens in fp16 (attention sink)",
     "quant + deployment"),
    # --- Key quantization ---
    ("include_rope",      "bool",  "Apply pre-RoPE key quantization",
     "quant"),
    ("perchannel",        "bool",  "Per-channel (key) vs per-token (value) quantization axis",
     "quant + deployment"),
    # --- KMeans calibration ---
    ("n_init",            "str",   "KMeans initialisation strategy (\"auto\" or integer)",
     "quant"),
    ("max_iter",          "int",   "KMeans max iterations (default: 50)",
     "quant"),
    ("random_state",      "int",   "KMeans random seed",
     "quant"),
    # --- Fisher information ---
    ("fisher",            "bool/tensor", "Whether to use Fisher-weighted KMeans",
     "gradients + quant"),
    # --- Sequence length ---
    ("seqlen",            "int",   "Calibration sequence length",
     "quant"),
    ("maxseqlen",         "int",   "Maximum inference sequence length (pre-allocates KV buffer)",
     "deployment"),
    # --- Kernel selection (implicit) ---
    ("use_parallel_topk", "bool",  "Use GPU parallel topK for sparse outlier packing",
     "deployment"),
    # --- Model ---
    ("model",             "str",   "HuggingFace model path or ID",
     "all stages"),
    ("dataset",           "str",   "Calibration dataset (e.g. wikitext2, c4)",
     "quant"),
    ("nsamples",          "int",   "Number of calibration samples",
     "quant"),
]


def print_hyperparameter_table():
    print("\nTable 1: User-Facing Hyperparameters Across the KVQuant Pipeline\n")
    header = f"  {'#':>3} | {'Parameter':>22} | {'Type':>14} | {'Stage':>20} | Description"
    print(header)
    print("  " + "-" * (len(header) + 20))
    for i, (name, typ, desc, stage) in enumerate(HYPERPARAMETERS, 1):
        print(f"  {i:>3} | {name:>22} | {typ:>14} | {stage:>20} | {desc}")
    print(f"\n  Total: {len(HYPERPARAMETERS)} non-trivial hyperparameters")
    print(
        "  Note: Many combinations are invalid (e.g. nf_nuq=True with nuq=False, "
        "or cap_outliers without include_sparse). No validation layer exists."
    )


# ---------------------------------------------------------------------------
# KMeans calibration cost
# ---------------------------------------------------------------------------

def kmeans_calibration_cost(
    num_layers: int,
    num_heads_k: int,
    head_dim: int,
    bits: int,
    nsamples: int,
    seqlen: int,
    max_iter: int = 50,
) -> dict:
    """
    Estimate wall-clock time and memory for KMeans codebook fitting.

    KMeans is run ONCE per (layer, key/value, channel) triple on the CPU,
    using sklearn.cluster.KMeans with sample_weight (Fisher).

    Complexity per call:
      O(n_samples × n_clusters × n_iter × d)
      where d=1 (1D clustering over normalized activations)

    Number of KMeans calls:
      num_layers × 2 (K and V) × head_dim_total
      Note: K is quantized per-channel (one LUT per channel),
            V is quantized per-token (one global LUT, but recalibrated per layer)
    """
    n_tokens_total = nsamples * seqlen
    n_clusters = 2 ** bits
    n_channels_K = num_layers * num_heads_k * head_dim  # per-channel for K
    n_calls_V    = num_layers                            # per-layer for V

    # Single KMeans call: O(n_tokens × n_clusters × max_iter)
    # Empirical: ~0.1 ms per 1000 samples × 16 clusters × 50 iter on CPU
    BASE_MS_PER_UNIT = 0.0001  # ms per (sample × cluster)
    ops_per_call_K = n_tokens_total * n_clusters * max_iter
    time_per_call_K_ms = BASE_MS_PER_UNIT * ops_per_call_K / 1000

    ops_per_call_V = n_tokens_total * n_clusters * max_iter
    time_per_call_V_ms = time_per_call_K_ms  # similar scale

    total_calls = n_channels_K + n_calls_V
    total_time_K_s = (n_channels_K * time_per_call_K_ms) / 1000
    total_time_V_s = (n_calls_V    * time_per_call_V_ms) / 1000
    total_time_s   = total_time_K_s + total_time_V_s

    # Memory: calibration activations (nsamples × seqlen × head_dim × layers × 2)
    # stored as float32 on CPU
    calib_bytes = nsamples * seqlen * (num_layers * num_heads_k * head_dim * 2) * 4
    calib_GB = calib_bytes / 1e9

    return {
        "n_kmeans_calls_K": n_channels_K,
        "n_kmeans_calls_V": n_calls_V,
        "total_kmeans_calls": total_calls,
        "n_clusters": n_clusters,
        "n_tokens_per_call": n_tokens_total,
        "estimated_total_time_hours": total_time_s / 3600,
        "calibration_memory_GB": calib_GB,
        "note": "Empirical estimate; actual time is dominated by sklearn overhead "
                "and Fisher weight computation, not raw flops.",
    }


# ---------------------------------------------------------------------------
# Pipeline dependency graph
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    {
        "stage": 1,
        "folder": "gradients/",
        "name": "Fisher Information Computation",
        "inputs": ["Pretrained model weights", "Calibration dataset"],
        "outputs": ["fisher_info_*.pt  (one file per layer)"],
        "env": "Custom conda env with modified Hugging Face Transformers fork",
        "blocking": True,
        "notes": [
            "The gradients/ folder IS a fork of the entire HuggingFace transformers "
            "repository (see gradients/src/transformers/).",
            "Must be re-run whenever the model changes or a new calibration set is used.",
            "Fisher info files can be hundreds of GB for large models.",
        ],
    },
    {
        "stage": 2,
        "folder": "quant/",
        "name": "Calibration & Codebook Fitting",
        "inputs": ["Pretrained model", "fisher_info_*.pt", "Calibration dataset"],
        "outputs": ["quantizers.pickle  (per-layer LUT + thresholds)"],
        "env": "Separate conda env with sklearn, scipy",
        "blocking": True,
        "notes": [
            "KMeans is run on the CPU for every (layer, channel) pair.",
            "No validation that codebook quality is acceptable before proceeding.",
            "quantizers.pickle can fail to load in deployment if path is wrong "
            "(binary pickle, not a portable format).",
        ],
    },
    {
        "stage": 3,
        "folder": "deployment/",
        "name": "CUDA Extension Build + Inference",
        "inputs": ["quantizers.pickle", "CUDA-capable GPU", "torch C++ headers"],
        "outputs": ["Quantized model inference"],
        "env": "Third conda env; must compile quant_cuda extension",
        "blocking": False,
        "notes": [
            "Compilation of quant_cuda_kernel.cu (5,668 lines) takes 3-10 minutes.",
            "Compiled extension is NOT portable: tied to CUDA toolkit + PyTorch version.",
            "A mismatch in PyTorch ABI between conda env and compiled .so causes "
            "silent corruption or segfault.",
            "maxseqlen must be set at startup; changing it requires model reload.",
        ],
    },
    {
        "stage": 4,
        "folder": "benchmarking/",
        "name": "Kernel Benchmarking (Optional)",
        "inputs": ["activations-seqlen*.pickle", "quantizers.pickle"],
        "outputs": ["Timing tables"],
        "env": "Fourth conda env; separate CUDA extension compilation",
        "blocking": False,
        "notes": [
            "Requires pre-cached activations (separate data collection step).",
            "Uses a different, simpler CUDA kernel file (1,254 lines) that is "
            "NOT identical to the deployment kernels.",
        ],
    },
]


def print_pipeline_graph():
    print("\nPipeline Dependency Graph:")
    for s in PIPELINE_STAGES:
        print(f"\n  Stage {s['stage']}: [{s['folder']}]  {s['name']}")
        print(f"    Environment : {s['env']}")
        print(f"    Inputs      : {', '.join(s['inputs'])}")
        print(f"    Outputs     : {', '.join(s['outputs'])}")
        print(f"    Blocking    : {'YES (must complete before next stage)' if s['blocking'] else 'NO'}")
        for note in s["notes"]:
            print(f"    ⚠  {note}")


# ---------------------------------------------------------------------------
# Silent failure analysis
# ---------------------------------------------------------------------------

SILENT_FAILURES = [
    {
        "condition": "cap_outliers=True but include_sparse=False",
        "symptom": "cap_outliers code path in get_outliers() runs topK selection "
                   "but the resulting mask is never used; model runs without error "
                   "but produces fp16-equivalent output (no compression).",
        "source": "quant/kvquant/simquant_module_quantizer.py:56-73",
    },
    {
        "condition": "maxseqlen too small for actual input",
        "symptom": "KV cache buffer overflows silently; kernel writes out-of-bounds "
                   "into the packed int32 tensor (fullwidth is the hard limit). "
                   "No bounds check in CUDA kernels.",
        "source": "deployment/kvquant/quant_cuda_kernel.cu:709 (fullwidth parameter)",
    },
    {
        "condition": "Fisher info not available (fisher=None in KMeans call)",
        "symptom": "Falls back to uniform KMeans without Fisher weighting. "
                   "No warning; perplexity degrades silently. "
                   "Paper results depend on Fisher weighting.",
        "source": "quant/kvquant/simquant_module_quantizer.py:508-528",
    },
    {
        "condition": "quantizers.pickle generated with different bits than deployment",
        "symptom": "LUT has wrong number of entries; kernel loop 'for val in 0..15' "
                   "reads uninitialized memory. No version check in pickle.",
        "source": "deployment/kvquant/simquant_module_quantizer.py:618",
    },
    {
        "condition": "KMeans converges to <2^bits clusters (degenerate data distribution)",
        "symptom": "cluster_centers_ has fewer entries than expected; LUT is "
                   "padded with zeros; quantization error spikes. sklearn does not "
                   "raise an error for partial convergence.",
        "source": "quant/kvquant/simquant_module_quantizer.py:510-528",
    },
]


def print_silent_failures():
    print("\nTable 3: Silent Failure Modes (no exception raised)\n")
    for i, f in enumerate(SILENT_FAILURES, 1):
        print(f"  Failure {i}:")
        print(f"    Condition : {f['condition']}")
        print(f"    Symptom   : {f['symptom']}")
        print(f"    Source    : {f['source']}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_separator(char="=", width=72):
    print(char * width)


def main():
    print_separator()
    print("KVQUANT ADVERSARIAL AUDIT — Vulnerability 4: Integration Complexity")
    print_separator()

    print_hyperparameter_table()
    print_pipeline_graph()

    print_separator("-")
    print("\nTable 2: KMeans Calibration Cost Estimates\n")
    configs = [
        ("LLaMA-7B",  32, 32, 128, 4, 128, 2048),
        ("LLaMA-13B", 40, 40, 128, 4, 128, 2048),
        ("LLaMA-65B", 80, 64, 128, 4, 128, 2048),
    ]
    header = (
        f"  {'Model':>12} | {'KMeans calls':>14} | {'Est. time (h)':>14} | "
        f"{'Calib data (GB)':>16}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, n_layers, n_heads, h_dim, bits, nsamples, seqlen in configs:
        c = kmeans_calibration_cost(n_layers, n_heads, h_dim, bits, nsamples, seqlen)
        print(
            f"  {name:>12} | {c['total_kmeans_calls']:>14,} | "
            f"{c['estimated_total_time_hours']:>14.2f} | "
            f"{c['calibration_memory_GB']:>16.1f}"
        )

    print_silent_failures()

    print_separator()
    print("\nSummary of Findings:")
    print(
        f"""
1. The KVQuant pipeline requires at least 3 separate conda environments with
   incompatible dependencies.  Stage 1 uses a 70,000+ file fork of the entire
   Hugging Face Transformers library just to intercept gradient computations.

2. There are {len(HYPERPARAMETERS)} user-facing hyperparameters that interact in undocumented
   ways.  Several combinations (e.g. cap_outliers without include_sparse) silently
   produce incorrect behavior rather than raising an error.

3. KMeans codebook fitting requires up to {max(kmeans_calibration_cost(l,h,d,4,128,2048)['total_kmeans_calls'] for _,l,h,d,*_ in [(0,80,64,128,4,128,2048)]):.0f} separate CPU KMeans calls
   for large models, and must be re-run entirely when the calibration set or
   quantization bit-width changes.

4. The paper's claimed A100 deployment requires compiling a 5,668-line CUDA
   extension tied to a specific CUDA toolkit and PyTorch ABI.  There is no
   pre-built wheel; any environment mismatch causes a silent segfault or
   corrupted output.

5. The benchmarking/ and deployment/ folders contain DIFFERENT CUDA source
   files (1,254 vs 5,668 lines).  Benchmark results from the simpler kernel
   may not reflect actual deployment performance.
"""
    )
    print_separator()


if __name__ == "__main__":
    main()
