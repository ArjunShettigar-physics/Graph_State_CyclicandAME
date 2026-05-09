"""
parallel_gpu.py  —  AME(18, 11) graph-state search
===================================================
Architecture
  • 1 MPI rank  →  1 GPU  (35 ranks = 35 Kepler GPUs)
  • CPU  : parameter decoding + vectorised matrix generation (NumPy)
  • GPU  : all C(18,9)=48 620 bipartition rank checks per matrix (Numba CUDA)

Total search space: (D-1) × D^8  =  10 × 11^8  ≈  2.14 billion matrices.
Each rank owns a contiguous 1/35 slice (~61 M matrices).

Kepler notes (CC 3.x)
  • Local arrays map to device-DRAM with L1/L2 caching — safe for 9×9 int32.
  • No FP16; all arithmetic is int32.
  • Fermat's little theorem used for modular inverse (avoids extended GCD).
"""

import sys
import numpy as np
from numba import cuda
import numba
from mpi4py import MPI
from itertools import combinations

# ── Problem constants ──────────────────────────────────────────────────────────
N        = 14                  # number of qudits
D        = 11                  # local dimension (prime)
M        = N // 2              # = 9  (half-system size)
NUM_FREE = (N - 2) // 2        # = 8  free parameters in row 1
D_POW    = D ** NUM_FREE       # 11^8 = 214 358 881

# GPU batch size — 1024 matrices per launch.
# flags buffer = 1024 × 48 620 × 1 B ≈ 47 MB: safe on Kepler VRAM.
BATCH = 262144

# ── MPI setup ─────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# One GPU per rank; wraps around if fewer GPUs than ranks on a node.
n_gpus = len(cuda.gpus)
cuda.select_device(rank % n_gpus)

# ── Precompute all C(18,9) = 48 620 bipartitions (done once, on every rank) ──
_K_list = list(combinations(range(N), M))
_all_K  = np.array(_K_list, dtype=np.int32)          # (48620, 9)  row indices
_all_C  = np.zeros_like(_all_K)                       # (48620, 9)  col indices
for _s, _row in enumerate(_K_list):
    _all_C[_s] = sorted(set(range(N)) - set(_row))

NUM_SUBSETS = _all_K.shape[0]   # 48 620

# Upload bipartition index tables to GPU — constant for entire run.
d_K = cuda.to_device(_all_K)
d_C = cuda.to_device(_all_C)


# ── CUDA kernel 1: extract sub-matrix + GF(p) rank ────────────────────────────
@cuda.jit
def ame_kernel(matrices, K_idx, C_idx, flags, p, n_mats, n_subs, m):
    """
    One thread  →  one (matrix_id, subset_id) pair.

    Extracts the m×m biadjacency sub-matrix A[K, C] into local memory and
    runs Gaussian elimination over GF(p).  Writes 1 to flags[mat_id, sub_id]
    iff the rank equals m (maximum), 0 otherwise.

    Parameters
    ----------
    matrices : (n_mats, N, N) int32   — batch of adjacency matrices
    K_idx    : (n_subs, m)   int32   — row indices for each bipartition
    C_idx    : (n_subs, m)   int32   — col indices for each bipartition
    flags    : (n_mats, n_subs) int8 — output: 1 = full rank
    p        : int32                 — prime modulus (= D)
    n_mats   : int32                 — actual matrices in this batch
    n_subs   : int32                 — number of bipartitions (48 620)
    m        : int32                 — half-system size (= M = 9)
    """
    tid = cuda.grid(1)
    if tid >= n_mats * n_subs:
        return

    mat_id = tid // n_subs
    sub_id = tid  % n_subs

    # ── Extract m×m sub-matrix into thread-local array (local memory / regs) ─
    # Kepler local memory: per-thread, cached in L1/L2 — fine for 9×9 int32.
    L = cuda.local.array((7, 7), dtype=numba.int32)
    for i in range(m):
        ri = K_idx[sub_id, i]
        for j in range(m):
            ci = C_idx[sub_id, j]
            L[i, j] = matrices[mat_id, ri, ci] % p

    # ── Gaussian elimination over GF(p) (reduced row echelon) ────────────────
    pp = p * p      # = 121;  guarantees (a - b + pp) % p ≥ 0 for b < p²
    rk = numba.int32(0)

    for c in range(m):
        # Pivot search in column c, rows rk … m-1
        piv = numba.int32(-1)
        for row in range(rk, m):
            if L[row, c] != 0:
                piv = row
                break
        if piv < 0:
            continue                # free column — rank does not increase

        # Row swap: bring pivot to row rk
        if piv != rk:
            for col in range(m):
                tmp         = L[rk,  col]
                L[rk,  col] = L[piv, col]
                L[piv, col] = tmp

        # Modular inverse of pivot via Fermat's little theorem:
        #   a^(p-1) ≡ 1 (mod p)  →  a^(-1) ≡ a^(p-2) (mod p)
        # For p=11, exponent = 9 = 1001₂  →  4 squarings.
        inv = numba.int32(1)
        b   = numba.int32(L[rk, c])
        e   = numba.int32(p - 2)
        while e:
            if e & 1:
                inv = (inv * b) % p
            b  = (b  * b) % p
            e >>= 1

        # Scale pivot row so pivot element becomes 1
        for col in range(m):
            L[rk, col] = (L[rk, col] * inv) % p

        # Eliminate pivot column in every other row
        for row in range(m):
            if row != rk and L[row, c] != 0:
                fac = numba.int32(L[row, c])
                for col in range(m):
                    # Add pp before % to keep the result non-negative
                    # (C-style % is truncated, not floored).
                    # fac * L[rk,col] < p² = 121 = pp, so adding pp suffices.
                    L[row, col] = (L[row, col] - fac * L[rk, col] + pp) % p

        rk += 1
        if rk == m:
            break   # already full rank — no need to continue

    # Write 1 iff full rank (matrix is entangled for this bipartition)
    flags[mat_id, sub_id] = numba.int8(rk == m)


# ── CUDA kernel 2: row-wise AND reduction ─────────────────────────────────────
@cuda.jit
def reduce_kernel(flags, mask, n_mats, n_subs):
    """
    One thread per matrix.
    mask[mat] = 1  iff  ALL flags[mat, 0…n_subs-1] == 1  (AME condition).
    Exits early on first failure for efficiency.
    """
    mat_id = cuda.grid(1)
    if mat_id >= n_mats:
        return
    ok = numba.int8(1)
    for s in range(n_subs):
        if flags[mat_id, s] == 0:
            ok = numba.int8(0)
            break
    mask[mat_id] = ok


# ── Index decode ──────────────────────────────────────────────────────────────
def decode_indices(idx_arr, d, nf):
    """
    Map global linear indices  →  (first_vals, values_batch).

    Layout:  global_idx = (first_val - 1) * d^nf  +  values_linear_idx
    first_val ∈ {1, …, d-1},  values_linear_idx encodes an nf-digit base-d number.
    """
    fv   = (idx_arr // D_POW).astype(np.int32) + 1   # 1 … d-1
    vlin = (idx_arr  % D_POW).astype(np.int64)

    vals = np.empty((len(idx_arr), nf), dtype=np.int32)
    for k in range(nf - 1, -1, -1):   # least-significant digit first
        vals[:, k] = vlin % d
        vlin //= d

    return fv, vals


# ── Matrix generation (vectorised NumPy — no Python loops over j) ─────────────
def build_matrices(fv, vals, n, d):
    """
    Vectorised construction of B adjacency matrices.

    fv   : (B,)      int32 — first-row constant
    vals : (B, nf)   int32 — free parameters for row 1
    Returns (B, n, n) int32  mod d.

    Key observations used for efficiency
    ─────────────────────────────────────
    • Row 0 upper triangle: all entries = fv  (one broadcast).
    • Row 1 upper triangle: symmetric pairs from vals  (num_free assignments).
    • Rows i ≥ 2 upper triangle: shift rule gives
          A[i, j] = A[i-1, j-1] = … = A[1, j-i+1]
      So the entire row i (for i ≥ 2) is a shifted slice of row 1 —
      filled with a single NumPy slice assignment (no inner j-loop).
    • Symmetrisation: one fancy-index assignment on the lower triangle.
    """
    B  = len(fv)
    nf = vals.shape[1]
    A  = np.zeros((B, n, n), dtype=np.int32)

    # Row 0: A[b, 0, 1:n] = fv[b]
    A[:, 0, 1:] = fv[:, None]

    # Row 1: symmetric pairs
    #   idx=0 → cols (2, n-1),  idx=1 → cols (3, n-2),  …
    for idx in range(nf):
        j  = 2 + idx
        jm = n - 1 - idx
        A[:, 1, j ] = vals[:, idx]
        A[:, 1, jm] = vals[:, idx]

    # Rows 2 … n-1: A[:, i, i+1 : n] = A[:, 1, 2 : n-i+1]
    # (slice lengths match: n-i-1 elements each side)
    for i in range(2, n):
        if n - i - 1 > 0:
            A[:, i, i + 1:] = A[:, 1, 2 : n - i + 1]

    # Symmetrise lower triangle
    iu = np.triu_indices(n, k=1)
    A[:, iu[1], iu[0]] = A[:, iu[0], iu[1]]

    return A % d


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    total    = (D - 1) * D_POW          # ≈ 2 143 588 810

    per_rank = (total + size - 1) // size
    my_start = rank * per_rank
    my_end   = min(my_start + per_rank, total)
    my_n     = max(my_end - my_start, 0)

    if rank == 0:
        print(f"{'─'*60}")
        print(f"  AME(18,11) GPU search")
        print(f"  Total matrices : {total:>20,}")
        print(f"  MPI ranks      : {size:>20,}")
        print(f"  Matrices/rank  : {per_rank:>20,}")
        print(f"  GPUs visible   : {n_gpus:>20,}")
        print(f"  Batch size     : {BATCH:>20,}")
        print(f"  Subsets/matrix : {NUM_SUBSETS:>20,}")
        print(f"{'─'*60}", flush=True)

    # Pre-allocate GPU buffers (reused across all batches — avoids allocation overhead)
    d_mats  = cuda.device_array((BATCH, N, N),         dtype=np.int32)
    d_flags = cuda.device_array((BATCH, NUM_SUBSETS),  dtype=np.int8)
    d_mask  = cuda.device_array((BATCH,),              dtype=np.int8)

    # Fixed kernel launch parameters (full-batch size)
    TPB      = 256
    BPG_kern = (BATCH * NUM_SUBSETS + TPB - 1) // TPB
    BPG_red  = (BATCH               + TPB - 1) // TPB

    ame_list = []                             # AME matrices found by this rank
    idx_arr  = np.arange(my_start, my_end, dtype=np.int64)

    for bs in range(0, my_n, BATCH):
        be = min(bs + BATCH, my_n)
        nb = be - bs                          # actual matrices in this batch

        # ── CPU: decode indices → parameters → matrices ───────────────────────
        fv, vals = decode_indices(idx_arr[bs:be], D, NUM_FREE)
        h_mats   = build_matrices(fv, vals, N, D)

        # ── Upload matrices to GPU ────────────────────────────────────────────
        d_mats[:nb].copy_to_device(h_mats)

        # Re-compute grid sizes for the (possibly smaller) last batch
        bpg_k = (nb * NUM_SUBSETS + TPB - 1) // TPB
        bpg_r = (nb               + TPB - 1) // TPB

        # ── GPU: compute all bipartition ranks ────────────────────────────────
        ame_kernel[bpg_k, TPB](
            d_mats, d_K, d_C, d_flags, D, nb, NUM_SUBSETS, M
        )

        # ── GPU: AND-reduce across subsets per matrix ─────────────────────────
        reduce_kernel[bpg_r, TPB](d_flags, d_mask, nb, NUM_SUBSETS)

        cuda.synchronize()

        # ── Download mask and collect hits ────────────────────────────────────
        h_mask = d_mask[:nb].copy_to_host()
        hits   = np.where(h_mask)[0]
        if hits.size:
            for h in hits:
                ame_list.append(h_mats[h].copy())
            print(f"  [rank {rank:02d}] *** {hits.size} AME matrix(ces) found! ***",
                  flush=True)

        # Progress (rank 0 only, every 500 batches)
        if rank == 0 and (bs // BATCH) % 500 == 0:
            pct = 100.0 * be / my_n
            print(f"  [rank 00] {be:>14,} / {my_n:,}  ({pct:6.2f}%)", flush=True)

    # ── Gather all results at rank 0 ─────────────────────────────────────────
    all_ame = comm.gather(ame_list, root=0)

    if rank == 0:
        combined = [m for sublist in all_ame for m in sublist]
        print(f"\n{'─'*60}")
        print(f"  DONE — found {len(combined)} AME(14,11) graph state(s).")
        if combined:
            out = np.stack(combined)
            np.save("ame_14_11.npy", out)
            print(f"  Saved {out.shape} array  →  ame_14_11.npy")
        print(f"{'─'*60}")


if __name__ == "__main__":
    main()
