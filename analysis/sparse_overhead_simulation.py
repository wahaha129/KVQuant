"""
Vulnerability 2: Dense-and-Sparse Quantization Overhead
=========================================================

KVQuant's dense-and-sparse scheme stores most KV activations in a packed
integer tensor (dense part) but keeps numerical outliers in a separate sparse
COO structure (rows/cols/vals).  Every time a new token is appended to the
KV cache, the C++ CUDA host code must:

  Phase 1 — quantize + detect outliers (GPU kernel launch)
  SYNC    — copy outlier_count from GPU to CPU (mandatory device-to-host sync)
  Phase 2 — scatter outliers into COO arrays (second GPU kernel launch)
  GROW    — extend sparse COO arrays with torch.cat() (O(nnz_so_far) copy)

This script simulates the memory and time overhead of that pipeline as a
function of context length and outlier sparsity, and compares it with a
hypothetical dense-only baseline.

Run with:
    python analysis/sparse_overhead_simulation.py
"""

import math
import time


# ---------------------------------------------------------------------------
# Model / hardware constants (LLaMA-7B on A100-80 GB)
# ---------------------------------------------------------------------------
NUM_LAYERS = 32
NUM_HEADS = 32
HEAD_DIM = 128            # per-head key/value dimension
BITS = 4                  # quantization bits
OUTLIER_RATE = 0.002      # 0.2 % of elements become outliers (paper §3.3)
FLOAT16_BYTES = 2
INT32_BYTES = 4
FLOAT32_BYTES = 4

# GPU <-> CPU synchronization latency (PCIe Gen4 x16, empirical)
D2H_SYNC_LATENCY_US = 5.0   # microseconds per scalar transfer + sync
KERNEL_LAUNCH_OVERHEAD_US = 2.0  # overhead per extra CUDA kernel launch


# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------

def dense_kv_memory_bytes(seq_len: int) -> dict:
    """
    Memory for the standard fp16 KV cache (no quantization, baseline).
    Layout: (layers, heads, head_dim, seq_len) × 2 (K and V)
    """
    elements = 2 * NUM_LAYERS * NUM_HEADS * HEAD_DIM * seq_len
    return {
        "fp16_bytes": elements * FLOAT16_BYTES,
        "fp16_MB": elements * FLOAT16_BYTES / 1e6,
    }


def kvquant_memory_bytes(seq_len: int, outlier_rate: float = OUTLIER_RATE) -> dict:
    """
    Memory for KVQuant's dense-and-sparse representation.

    Dense part:  packed int tensors  — (heads, head_dim//(32//bits), seq_len)
                 lookup table (LUT)   — (heads, head_dim, 2^bits) × float32

    Sparse part: COO outlier values  — (num_outliers,) × float16
                 COO outlier indices — (num_outliers,) × int32
                 CSR row-pointer     — (seq_len+1,)    × int32   (per-layer)
                 start_rows helper   — (num_threads,)  × int32

    This is computed for a single Key or Value cache; multiply by 2×layers.
    """
    total_elements = NUM_HEADS * HEAD_DIM * seq_len

    # Dense quantized part
    packing_ratio = 32 // BITS          # 4-bit: 8 values per int32
    dense_ints = math.ceil(total_elements / packing_ratio)
    dense_bytes = dense_ints * INT32_BYTES

    # LUT (one per head × head_dim channel; 2^bits entries each)
    lut_entries = NUM_HEADS * HEAD_DIM * (2 ** BITS)
    lut_bytes = lut_entries * FLOAT32_BYTES

    # Sparse COO part
    num_outliers = int(total_elements * outlier_rate)
    coo_val_bytes = num_outliers * FLOAT16_BYTES
    coo_idx_bytes = num_outliers * INT32_BYTES
    # CSR row-pointer: one entry per token + 1  (grows linearly)
    csr_rowptr_bytes = (seq_len + 1) * INT32_BYTES
    # start_rows helper: ceil(nnz / 10) entries
    start_rows_bytes = math.ceil(num_outliers / 10) * INT32_BYTES

    sparse_bytes = coo_val_bytes + coo_idx_bytes + csr_rowptr_bytes + start_rows_bytes

    total_one_kv = dense_bytes + lut_bytes + sparse_bytes
    total_both_kv = total_one_kv * 2            # K and V
    total_all_layers = total_both_kv * NUM_LAYERS

    return {
        "dense_packed_MB": dense_bytes / 1e6,
        "lut_MB": lut_bytes / 1e6,
        "sparse_COO_MB": sparse_bytes / 1e6,
        "total_single_KV_MB": total_one_kv / 1e6,
        "total_all_layers_MB": total_all_layers / 1e6,
        "num_outliers": num_outliers,
        "outlier_rate": outlier_rate,
    }


# ---------------------------------------------------------------------------
# Latency model for token-append overhead
# ---------------------------------------------------------------------------

def append_overhead_per_token_us(seq_len: int, outlier_rate: float = OUTLIER_RATE) -> dict:
    """
    Estimate the latency overhead of one dense-and-sparse token append
    relative to a simple fp16 token append.

    Key costs that do NOT exist in the baseline:
      (a) Extra kernel launch for Phase 2 (outlier scatter)
      (b) Mandatory CPU-GPU synchronization to read outlier_count
      (c) torch.cat() copy of growing COO arrays — O(nnz_current) copy
    """
    num_outliers_so_far = int(NUM_HEADS * HEAD_DIM * seq_len * outlier_rate)

    # (a) Phase 2 extra kernel launch
    extra_launch_us = KERNEL_LAUNCH_OVERHEAD_US

    # (b) CPU-GPU sync: D2H copy of a single int32 scalar
    sync_us = D2H_SYNC_LATENCY_US

    # (c) torch.cat() to grow sparse COO arrays: proportional to current nnz
    #     The COO tensors are CUDA tensors; torch::cat copies on the GPU.
    #     A100 GPU device-to-device bandwidth: ~2 TB/s
    COPY_BANDWIDTH_GB_S = 2000.0   # A100 HBM2e peak bandwidth
    coo_bytes_to_copy = num_outliers_so_far * (FLOAT16_BYTES + INT32_BYTES)
    cat_us = (coo_bytes_to_copy / (COPY_BANDWIDTH_GB_S * 1e9)) * 1e6

    total_overhead_us = extra_launch_us + sync_us + cat_us

    return {
        "extra_kernel_launch_us": extra_launch_us,
        "cpu_gpu_sync_us": sync_us,
        "coo_concat_copy_us": cat_us,
        "total_overhead_per_token_us": total_overhead_us,
    }


def cumulative_append_overhead_s(max_seq_len: int) -> float:
    """Total cumulative overhead (seconds) for building a full KV cache."""
    total_us = 0.0
    # Step through the sequence length in strides for efficiency
    stride = max(1, max_seq_len // 1000)
    for seq_len in range(stride, max_seq_len + 1, stride):
        ov = append_overhead_per_token_us(seq_len)
        total_us += ov["total_overhead_per_token_us"] * stride
    return total_us / 1e6   # convert to seconds


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------

def print_separator(char="=", width=72):
    print(char * width)


def main():
    print_separator()
    print("KVQUANT ADVERSARIAL AUDIT — Vulnerability 2: Dense-and-Sparse Overhead")
    print_separator()

    # -----------------------------------------------------------------------
    # Table 1: Memory comparison
    # -----------------------------------------------------------------------
    print("\nTable 1: Memory (GB) vs. Context Length — Baseline fp16 vs. KVQuant")
    print(f"  (Model: LLaMA-7B, {NUM_LAYERS} layers, {NUM_HEADS} heads, "
          f"head_dim={HEAD_DIM}, bits={BITS})")
    print(f"  Outlier rate: {OUTLIER_RATE*100:.1f}%\n")

    header = f"  {'SeqLen':>10} | {'fp16 (GB)':>12} | {'KVQuant (GB)':>14} | {'Overhead':>10} | {'Outliers':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for seq_len in [1_000, 10_000, 100_000, 250_000, 500_000, 1_000_000]:
        fp16 = dense_kv_memory_bytes(seq_len)["fp16_bytes"] / 1e9
        kv = kvquant_memory_bytes(seq_len)
        kvq_total = kv["total_all_layers_MB"] / 1e3
        overhead_ratio = kvq_total / fp16
        print(
            f"  {seq_len:>10,} | {fp16:>12.3f} | {kvq_total:>14.3f} | "
            f"{overhead_ratio:>9.2f}x | {kv['num_outliers']:>12,}"
        )

    # -----------------------------------------------------------------------
    # Table 2: Per-token append latency overhead vs. sequence length
    # -----------------------------------------------------------------------
    print("\nTable 2: Per-token append overhead (µs) due to dense-and-sparse mechanism")
    print(f"  (For a single model layer; scale ×{NUM_LAYERS} for full model)\n")

    header2 = (
        f"  {'SeqLen':>10} | {'ExtraKernel(µs)':>16} | {'CPUsync(µs)':>12} | "
        f"{'CatCopy(µs)':>12} | {'Total(µs)':>10}"
    )
    print(header2)
    print("  " + "-" * (len(header2) - 2))

    for seq_len in [1_000, 10_000, 100_000, 500_000, 1_000_000]:
        ov = append_overhead_per_token_us(seq_len)
        print(
            f"  {seq_len:>10,} | {ov['extra_kernel_launch_us']:>16.1f} | "
            f"{ov['cpu_gpu_sync_us']:>12.1f} | "
            f"{ov['coo_concat_copy_us']:>12.3f} | "
            f"{ov['total_overhead_per_token_us']:>10.3f}"
        )

    # -----------------------------------------------------------------------
    # Table 3: Cumulative overhead for full-context prefill
    # -----------------------------------------------------------------------
    print("\nTable 3: Cumulative append overhead (seconds) to build the KV cache")
    print(f"  (Per layer × {NUM_LAYERS} layers for full model)\n")

    header3 = f"  {'MaxSeqLen':>12} | {'Per-layer (s)':>14} | {'Full model (s)':>15}"
    print(header3)
    print("  " + "-" * (len(header3) - 2))

    for seq_len in [10_000, 100_000, 500_000, 1_000_000]:
        per_layer_s = cumulative_append_overhead_s(seq_len)
        full_model_s = per_layer_s * NUM_LAYERS
        print(f"  {seq_len:>12,} | {per_layer_s:>14.3f} | {full_model_s:>15.3f}")

    # -----------------------------------------------------------------------
    # Critical code paths from source
    # -----------------------------------------------------------------------
    print_separator("-")
    print("\nCritical code paths verified in deployment/kvquant/quant_cuda_kernel.cu:\n")
    print(
        "  Line 745-748 (vecquant4appendvecKsparseorig_cuda):"
    )
    print(
        "    torch::Tensor hostcount = outlier_count.to(torch::kCPU);  // << SYNC BARRIER\n"
        "    int* count = hostcount.data_ptr<int>();\n"
        "    ...allocate dst_indices and dst_values based on intcount...\n"
        "    VecQuant4AppendVecKSparse2Orig<<<...>>>(...);  // << SECOND KERNEL\n"
    )
    print(
        "  Lines 796-809 (sparse COO growth via torch::cat):"
    )
    print(
        "    row2 = torch::cat({row, dst_row_cuda}, 0);\n"
        "    col2 = torch::cat({col, dst_indices},  0);  // copies ENTIRE history\n"
        "    val2 = torch::cat({val, dst_values},   0);\n"
    )
    print_separator()
    print("\nSummary of Findings:")
    print(
        """
1. Every new KV token triggers a mandatory GPU-to-CPU synchronization to
   read the dynamic outlier count.  This stalls the GPU pipeline and prevents
   asynchronous execution, contradicting the paper's implicit assumption of
   "efficient" online quantization.

2. The sparse COO arrays are grown via torch::cat(), which copies ALL prior
   outlier data on every token.  At 1M-token context and 0.2% outlier rate,
   this cumulative copy cost reaches O(seqlen²) total bytes transferred.

3. The two-phase kernel design (quantize → sync → scatter) doubles the number
   of CUDA kernel launches compared to a single-pass approach, increasing
   driver-overhead and reducing GPU occupancy at short sequence lengths.

4. The paper reports memory savings (up to 12× at 4-bit) but does not report
   the wall-clock latency cost of building the sparse KV cache during prefill,
   which can exceed several seconds at 1M context even on an A100.
"""
    )
    print_separator()


if __name__ == "__main__":
    main()
