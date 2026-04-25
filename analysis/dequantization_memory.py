"""
Vulnerability 3: Dequantization Bottleneck — LUT Cache Pressure
=================================================================

KVQuant's NUQ (Non-Uniform Quantization) represents each quantized value as
a 4-bit (or 2/3-bit) index into a per-channel floating-point lookup table
(LUT).  During the Q*K^T attention matrix-vector product, the kernel must:

  1. Load a LUT of shape (num_heads × head_dim × 2^bits) fp32 values into
     GPU shared memory.
  2. Dequantize each packed int32 by looking up 8 consecutive LUT entries.
  3. Compute cos/sin/pow trigonometric values per decoded element (RoPE).

This script quantifies:
  (a) LUT memory footprint vs. GPU shared-memory and L1-cache capacity.
  (b) Effective "dequantization arithmetic intensity" vs. standard fp16.
  (c) Shared-memory occupancy reduction caused by the large LUT allocation.

Run with:
    python analysis/dequantization_memory.py
"""

import math


# ---------------------------------------------------------------------------
# Hardware constants (NVIDIA A100-80GB, SM count = 108)
# ---------------------------------------------------------------------------
SM_SHARED_MEM_BYTES = 164 * 1024   # 164 KB configurable shared mem per SM
SM_L1_CACHE_BYTES   =  32 * 1024   # ~32 KB L1 cache per SM
SM_REGISTERS        = 65536         # 65536 32-bit registers per SM
MAX_THREADS_PER_SM  = 2048          # hardware limit
WARP_SIZE           = 32

# Kernel block configuration (from quant_cuda_kernel.cu)
BLOCKWIDTH   = 128
BLOCKHEIGHT4 =  16    # for 4-bit packing

# LUT configuration
LUT_ENTRIES_PER_CHANNEL = 16        # 2^4 for 4-bit NUQ


# ---------------------------------------------------------------------------
# LUT footprint analysis
# ---------------------------------------------------------------------------

def lut_shared_memory_bytes(num_heads: int, head_dim: int, bits: int) -> dict:
    """
    Compute the shared-memory allocation for the LUT inside the fused kernel.

    From quant_cuda_kernel.cu (line ~3067):
        __shared__ float deq2[17][BLOCKWIDTH];

    The LUT is loaded as deq2[0..15][threadIdx.x], where each thread handles
    one channel.  deq2[16] holds precomputed RoPE theta values.
    """
    # In the fused kernel, BLOCKWIDTH threads each load their own LUT row
    # deq2 has 17 rows × BLOCKWIDTH columns × 4 bytes
    deq2_bytes = 17 * BLOCKWIDTH * 4     # float32

    # Additional shared arrays (blockvec, blockvec2)
    blockvec_bytes = 2 * BLOCKWIDTH * 4  # two vectors

    total_shared_per_block = deq2_bytes + blockvec_bytes

    # Full LUT stored in global memory (loaded to shared per block)
    full_lut_global_bytes = num_heads * head_dim * (2 ** bits) * 4
    full_lut_global_MB = full_lut_global_bytes / 1e6

    threads_per_block = BLOCKWIDTH

    # Number of thread blocks that fit on one SM simultaneously
    # (limited by BOTH shared memory and thread count)
    blocks_by_smem = SM_SHARED_MEM_BYTES // total_shared_per_block
    blocks_by_threads = MAX_THREADS_PER_SM // threads_per_block
    blocks_per_sm = min(blocks_by_smem, blocks_by_threads)

    # Active warps per SM
    active_warps = blocks_per_sm * (threads_per_block // WARP_SIZE)
    theoretical_max_warps = MAX_THREADS_PER_SM // WARP_SIZE
    occupancy = active_warps / theoretical_max_warps

    # Shared memory used by active blocks
    smem_used_bytes = blocks_per_sm * total_shared_per_block
    smem_fraction = smem_used_bytes / SM_SHARED_MEM_BYTES

    return {
        "deq2_shared_bytes": deq2_bytes,
        "blockvec_shared_bytes": blockvec_bytes,
        "total_shared_per_block_bytes": total_shared_per_block,
        "total_shared_per_block_KB": total_shared_per_block / 1024,
        "max_blocks_per_SM": blocks_per_sm,
        "smem_used_fraction": smem_fraction,
        "active_warps_per_SM": active_warps,
        "theoretical_max_warps": theoretical_max_warps,
        "sm_occupancy": occupancy,
        "full_lut_global_MB": full_lut_global_MB,
        "l1_cache_fit": full_lut_global_bytes <= SM_L1_CACHE_BYTES,
    }


# ---------------------------------------------------------------------------
# Arithmetic intensity comparison
# ---------------------------------------------------------------------------

def arithmetic_intensity_fp16(seq_len: int, head_dim: int) -> dict:
    """
    Standard fp16 dense matrix-vector product: Q * K^T
    Flops: 2 × head_dim × seq_len  (multiply-add)
    Bytes loaded: head_dim × seq_len × 2  (fp16 K cache)
    """
    flops = 2 * head_dim * seq_len
    bytes_loaded = head_dim * seq_len * 2        # fp16
    intensity = flops / bytes_loaded
    return {"flops": flops, "bytes": bytes_loaded, "intensity_flops_per_byte": intensity}


def arithmetic_intensity_nuq4(seq_len: int, head_dim: int) -> dict:
    """
    NUQ 4-bit MatVec: Q * K_quantized^T with LUT dequantization

    Flops per output element:
      - Unpack 1 int32 → 8 x 4-bit indices       (8 shift+mask ops)
      - 8 LUT lookups (shared memory float load)
      - 8 multiply-accumulate (float32)
      - 2 × cosf + 2 × sinf + 1 × powf per element (RoPE, from kernel lines 3125-3186)
      → ~8 + 8 + 8 + (8 × 5 trig) = 64 "ops" per 8 elements

    Bytes loaded:
      - Packed int32 K cache: head_dim × seq_len / (32/4) × 4 bytes
      - LUT (shared mem; count as L1 pressure, ~34 KB per block):
        approximated as LUT_entries × 4 bytes per output block
    """
    packed_int32_per_row = head_dim // 8         # 4-bit, 8 per int32
    packed_bytes = packed_int32_per_row * seq_len * 4

    # LUT: each block reads the full LUT row for its head
    lut_reads_bytes = head_dim * LUT_ENTRIES_PER_CHANNEL * 4  # once per block

    total_bytes = packed_bytes + lut_reads_bytes

    # Flops: unpack + lut + mac + trig (cosf ≈ 20 FLOP equivalent on GPU)
    TRIG_FLOP_EQUIV = 20
    mac_flops  = 2 * head_dim * seq_len                     # multiply-add
    trig_flops = 2 * head_dim * seq_len * TRIG_FLOP_EQUIV   # cos + sin per element
    total_flops = mac_flops + trig_flops

    intensity = total_flops / total_bytes

    return {
        "mac_flops": mac_flops,
        "trig_flops_equiv": trig_flops,
        "total_flops_equiv": total_flops,
        "packed_bytes_loaded": packed_bytes,
        "lut_bytes_loaded": lut_reads_bytes,
        "total_bytes": total_bytes,
        "intensity_flops_per_byte": intensity,
    }


def speedup_from_lower_bw(seq_len: int, head_dim: int) -> float:
    """
    Theoretical speedup from bandwidth reduction alone (ignoring dequant cost).
    fp16 bytes / NUQ-4 packed bytes.
    """
    fp16_bw = head_dim * seq_len * 2
    nuq_bw  = (head_dim * seq_len) // 2  # 4-bit = 0.5 bytes per element
    return fp16_bw / nuq_bw              # = 4× bandwidth reduction


# ---------------------------------------------------------------------------
# Trig cost breakdown per generated token
# ---------------------------------------------------------------------------

def trig_cost_per_token(seq_len: int, num_layers: int, num_heads: int,
                        head_dim: int) -> dict:
    """
    Total transcendental function evaluations needed for one generated token
    across all layers and heads (Q*K^T attention with inline RoPE dequant).

    From the kernel: cosf + sinf called once per 4-bit decoded element,
    for every column (seq position) in the attention computation.
    """
    trig_per_head = 2 * head_dim * seq_len   # cosf + sinf per element
    trig_total = trig_per_head * num_heads * num_layers
    return {
        "trig_per_head": trig_per_head,
        "trig_all_heads_all_layers": trig_total,
        "note": (
            f"At 1M context: {2 * head_dim * 1_000_000 * num_heads * num_layers:,.0f} "
            "transcendental evaluations per token generated — none precomputed."
        ),
    }


def print_separator(char="=", width=72):
    print(char * width)


def main():
    # LLaMA-7B parameters
    models = [
        ("LLaMA-7B",  32, 128, 32),
        ("LLaMA-13B", 40, 128, 40),
        ("LLaMA-65B", 64, 128, 80),
    ]

    print_separator()
    print("KVQUANT ADVERSARIAL AUDIT — Vulnerability 3: Dequantization Bottleneck")
    print_separator()

    # -----------------------------------------------------------------------
    # Table 1: LUT shared-memory pressure per model
    # -----------------------------------------------------------------------
    print("\nTable 1: LUT Shared-Memory Allocation and SM Occupancy (4-bit NUQ)")
    print(
        f"  A100 SM resources: {SM_SHARED_MEM_BYTES//1024} KB shared mem, "
        f"{SM_L1_CACHE_BYTES//1024} KB L1 cache, {MAX_THREADS_PER_SM} max threads\n"
    )

    header = (
        f"  {'Model':>12} | {'SharedMem/blk(KB)':>18} | {'Blocks/SM':>10} | "
        f"{'Occupancy':>10} | {'SMEM used%':>10} | {'GlobalLUT(KB)':>14} | {'Fits L1?':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, n_heads, h_dim, n_layers in models:
        info = lut_shared_memory_bytes(n_heads, h_dim, bits=4)
        print(
            f"  {name:>12} | {info['total_shared_per_block_KB']:>18.1f} | "
            f"{info['max_blocks_per_SM']:>10} | "
            f"{info['sm_occupancy']:>10.1%} | "
            f"{info['smem_used_fraction']:>10.1%} | "
            f"{info['full_lut_global_MB']*1000:>14.1f} | "
            f"{'YES' if info['l1_cache_fit'] else 'NO':>8}"
        )

    # -----------------------------------------------------------------------
    # Table 2: Arithmetic intensity comparison
    # -----------------------------------------------------------------------
    print("\nTable 2: Arithmetic Intensity — fp16 baseline vs NUQ-4bit + RoPE inline")
    print(f"  (LLaMA-7B, head_dim=128, TRIG_FLOP_EQUIV=20 FLOPs per transcendental)\n")

    header2 = (
        f"  {'SeqLen':>10} | {'fp16 (FLOP/B)':>14} | {'NUQ-4+RoPE':>12} | "
        f"{'BW saving':>10} | {'Trig overhead':>15}"
    )
    print(header2)
    print("  " + "-" * (len(header2) - 2))

    for seq_len in [1_000, 10_000, 100_000, 1_000_000]:
        fp16 = arithmetic_intensity_fp16(seq_len, 128)
        nuq  = arithmetic_intensity_nuq4(seq_len, 128)
        bw_save = speedup_from_lower_bw(seq_len, 128)
        trig_fraction = nuq["trig_flops_equiv"] / nuq["total_flops_equiv"]
        print(
            f"  {seq_len:>10,} | {fp16['intensity_flops_per_byte']:>14.2f} | "
            f"{nuq['intensity_flops_per_byte']:>12.2f} | "
            f"{bw_save:>10.1f}× | "
            f"{trig_fraction:>15.1%}"
        )

    # -----------------------------------------------------------------------
    # Table 3: Trig evaluations per generated token
    # -----------------------------------------------------------------------
    print("\nTable 3: Transcendental function calls per token generated (Q*K^T)")
    print("  (No precomputed sine/cosine table; recomputed per-element per query)\n")

    header3 = f"  {'SeqLen':>12} | {'Trig evals (LLaMA-7B)':>24} | {'Trig evals (LLaMA-65B)':>26}"
    print(header3)
    print("  " + "-" * (len(header3) - 2))

    for seq_len in [10_000, 100_000, 500_000, 1_000_000]:
        t7b  = trig_cost_per_token(seq_len, 32, 32, 128)
        t65b = trig_cost_per_token(seq_len, 80, 64, 128)
        print(
            f"  {seq_len:>12,} | {t7b['trig_all_heads_all_layers']:>24,} | "
            f"{t65b['trig_all_heads_all_layers']:>26,}"
        )

    # -----------------------------------------------------------------------
    # Source code reference
    # -----------------------------------------------------------------------
    print_separator("-")
    print("\nVerified code paths in deployment/kvquant/quant_cuda_kernel.cu:\n")
    print(
        "  Line 3067 (VecQuant4MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt):\n"
        "    __shared__ float deq2[17][BLOCKWIDTH];  // 17×128×4 = 8 704 B / block\n"
        "    for (int val = 0; val < 16; val++) {\n"
        "        deq2[val][off] = lookup_table[lut_index];  // 16 global→shared loads\n"
        "    }\n"
        "    deq2[16][off] = powf(rope_theta, ...);  // RoPE theta precompute (shared)\n"
    )
    print(
        "  Lines 3125-3199 (inner decode loop, unrolled 8× per int32):\n"
        "    c = cosf(theta * pos);   // transcendental — per decoded element\n"
        "    s = sinf(theta * pos);   // transcendental — per decoded element\n"
        "    res += tmp1 * c * blockvec[k];\n"
        "    res += sign * tmp1 * s * blockvec2[k];\n"
    )

    print_separator()
    print("\nSummary of Findings:")
    print(
        """
1. The fused NUQ+RoPE kernel allocates 9.5 KB of shared memory per thread
   block for the LUT (8.7 KB) plus input vectors (0.8 KB).  On an A100 with
   2048 max threads per SM, occupancy is thread-limited (not shared-mem
   limited), reaching 100%.  However, the LUT consumes 93% of the per-block
   shared memory allocation, leaving almost no headroom for register spilling
   or future kernel features without sacrificing occupancy.

2. The full per-layer LUT (256 KB for LLaMA-7B at 4-bit) is 8× larger than
   the A100 L1 cache (32 KB per SM).  Each attention block must re-load the
   LUT from L2/global memory on every block launch, causing repeated L2 cache
   pressure that negates part of the bandwidth saving from 4-bit packing.

3. cosf() and sinf() are called inline per decoded 4-bit element rather than
   precomputed into a table.  At 1M-token context, this produces over
   2.6 × 10^11 transcendental evaluations per token for LLaMA-7B —
   a compute cost that the paper does not report separately.

4. The net arithmetic intensity of NUQ-4+RoPE is NOT simply 4× higher than
   fp16: the trig overhead contributes >95% of the total FLOP count at long
   contexts, making the kernel severely compute-bound rather than bandwidth-
   bound and effectively eroding the memory-bandwidth-reduction argument.
"""
    )
    print_separator()


if __name__ == "__main__":
    main()
