"""
parallel_gpu3.py  —  AME(2n, p) graph-state search  [canonical form]
=====================================================================
Architecture
  • 1 MPI rank  →  1 GPU
  • CPU  : canonical vector generation + vectorised matrix construction (NumPy)
  • GPU  : bipartition rank checks (Numba CUDA)

Changes from previous version
──────────────────────────────
  1. Row 0 upper triangle is fixed to 1 (was swept over 1..p-1).

  2. Only bipartitions where qudit 0 is in the ROW set K are checked.
     Since the matrix is symmetric, rank(A[K,C]) = rank(A[C,K]), so
     the (B,A) case is redundant. Fixing 0 ∈ K gives exactly C(2n-1, n-1)
     bipartitions instead of C(2n, n).
     For N=14: C(13,6) = 1 716  (down from 3 432).

  3. Row 1 canonical form — first non-zero free parameter is always 1.
     Free params: a1, a2, …, a(n-1)  placed as
       row 1 = [-, 0, a1, a2, …, a(n-1), a(n-1), …, a2, a1]
     Enumeration by "leading-one block":
       block k  (0-indexed): a1=…=ak=0, a(k+1)=1, rest free ∈ {0..p-1}
     Total canonical vectors = (p^(n-1) - 1) / (p - 1).
     For N=14, p=11: (11^6 - 1)/10 = 177 156  (down from 11^6 = 1 771 561).

Combined reduction vs. original:
  Original : (p-1) × p^(n-1) ≈ 2.14 B   (N=14, p=11)
  New      : (p^(n-1)-1)/(p-1) = 177 156
  Speed-up : ~12 000×

Inputs
──────
  N  (= 2n) : total number of qudits     — set below
  D  (= p)  : local dimension (prime)    — set below
"""

import numpy as np
from numba import cuda
import numba
from mpi4py import MPI
from itertools import combinations, product

# ══════════════════════════════════════════════════════════════════════════════
#  Problem constants  ← only these two lines need to change for a different run
# ══════════════════════════════════════════════════════════════════════════════
N = 20          # total qudits  (must be even)
D = 7        # local dimension  (must be prime)
# ══════════════════════════════════════════════════════════════════════════════

M        = N // 2        # half-system size  (= n)
NUM_FREE = M - 1         # free params in row 1  (= n-1)

BATCH = 65536

# ── MPI + GPU setup ───────────────────────────────────────────────────────────
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
size   = comm.Get_size()
n_gpus = len(cuda.gpus)
cuda.select_device(rank % n_gpus)


# ── Change 3: generate all canonical row-1 parameter vectors ─────────────────
def generate_canonical_rows(nf, p):
    """
    Enumerate all (a1, …, anf) over GF(p) whose first non-zero entry is 1.

    Block k  (k = 0 … nf-1):
        a1 = … = ak = 0,  a(k+1) = 1,  a(k+2) … anf  ∈ {0 … p-1}

    Total = sum_{k=0}^{nf-1} p^(nf-k-1) = (p^nf - 1) / (p - 1).
    """
    rows = []
    for k in range(nf):                          # k = index of leading one
        n_tail = nf - k - 1                      # free positions after the 1
        for tail in product(range(p), repeat=n_tail):
            rows.append([0] * k + [1] + list(tail))
    return np.array(rows, dtype=np.int32)        # (TOTAL, nf)

_all_vals = generate_canonical_rows(NUM_FREE, D)
TOTAL     = len(_all_vals)   # (D^NUM_FREE - 1) / (D - 1)  =  177 156  for N=14,D=11


# ── Change 2: bipartitions with 0 in the row set K only ──────────────────────
# rank(A[K,C]) == rank(A[C,K]) by symmetry  →  only one orientation needed.
# Fixing 0 ∈ K: choose remaining M-1 row qudits from {1, …, N-1}  → C(N-1, M-1).
_K_list = [k for k in combinations(range(N), M) if 0 in k]
_all_K  = np.array(_K_list, dtype=np.int32)           # (C(N-1,M-1), M)
_all_C  = np.zeros_like(_all_K)
for _s, _row in enumerate(_K_list):
    _all_C[_s] = sorted(set(range(N)) - set(_row))

NUM_SUBSETS = _all_K.shape[0]   # C(N-1, M-1) = C(13, 6) = 1 716  for N=14

# Upload bipartition tables to GPU — constant for entire run
d_K = cuda.to_device(_all_K)
d_C = cuda.to_device(_all_C)


# ── CUDA kernel 1: extract M×M sub-matrix and compute GF(p) rank ─────────────
@cuda.jit
def ame_kernel(matrices, K_idx, C_idx, flags, p, n_mats, n_subs, m):
    """
    One thread → one (matrix_id, subset_id) pair.

    Extracts A[K, C] into a thread-local M×M array and runs Gaussian
    elimination over GF(p).  Sets flags[mat_id, sub_id] = 1 iff rank = m.

    NOTE: cuda.local.array size must be a compile-time constant.
          M is a module-level Python int so Numba accepts it as such.
    """
    tid = cuda.grid(1)
    if tid >= n_mats * n_subs:
        return

    mat_id = tid // n_subs
    sub_id = tid  % n_subs

    # Thread-local M×M work array (cached in L1/L2 on Kepler)
    L = cuda.local.array((M, M), dtype=numba.int32)
    for i in range(m):
        ri = K_idx[sub_id, i]
        for j in range(m):
            ci = C_idx[sub_id, j]
            L[i, j] = matrices[mat_id, ri, ci] % p

    # Gaussian elimination over GF(p)
    # pp = p² guarantees (a - b*c + pp) % p ≥ 0 when b,c < p
    pp = p * p
    rk = numba.int32(0)

    for c in range(m):
        # Find pivot in column c at or below current row rk
        piv = numba.int32(-1)
        for row in range(rk, m):
            if L[row, c] != 0:
                piv = row
                break
        if piv < 0:
            continue                    # free column — rank stays the same

        # Bring pivot row up to position rk
        if piv != rk:
            for col in range(m):
                tmp         = L[rk,  col]
                L[rk,  col] = L[piv, col]
                L[piv, col] = tmp

        # Modular inverse via Fermat: a^(p-2) mod p
        inv = numba.int32(1)
        b   = numba.int32(L[rk, c])
        e   = numba.int32(p - 2)
        while e:
            if e & 1:
                inv = (inv * b) % p
            b  = (b * b) % p
            e >>= 1

        # Scale pivot row → leading 1
        for col in range(m):
            L[rk, col] = (L[rk, col] * inv) % p

        # Eliminate all other rows in this column
        for row in range(m):
            if row != rk and L[row, c] != 0:
                fac = numba.int32(L[row, c])
                for col in range(m):
                    L[row, col] = (L[row, col] - fac * L[rk, col] + pp) % p

        rk += 1
        if rk == m:
            break       # already full rank — no need to continue

    flags[mat_id, sub_id] = numba.int8(rk == m)


# ── CUDA kernel 2: AND-reduce flags → single pass/fail bit per matrix ─────────
@cuda.jit
def reduce_kernel(flags, mask, n_mats, n_subs):
    """
    One thread per matrix.
    mask[mat] = 1  iff  ALL flags[mat, 0…n_subs-1] == 1  (AME condition).
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


# ── Matrix construction (vectorised NumPy) ────────────────────────────────────
def build_matrices(vals, n, d):
    """
    Construct B adjacency matrices from canonical row-1 parameter vectors.

    vals : (B, nf) int32  — canonical free parameters a1 … a(n-1)

    Structure (upper triangle, then symmetrise):
      Row 0 : all 1s                              [Change 1 — fixed fv=1]
      Row 1 : [-, 0, a1, a2, …, a(n-1), a(n-1), …, a2, a1]
      Row i≥2: Toeplitz shift — A[:, i, i+1:] = A[:, 1, 2:n-i+1]
    """
    B  = len(vals)
    nf = vals.shape[1]          # = M - 1 = n - 1
    A  = np.zeros((B, n, n), dtype=np.int32)

    # ── Row 0: fixed to all 1s (Change 1) ─────────────────────────────────────
    A[:, 0, 1:] = 1

    # ── Row 1: symmetric placement of free parameters ─────────────────────────
    #   idx  →  positions (2+idx) and (n-1-idx)
    #   idx=0 → (2, n-1),  idx=1 → (3, n-2),  …,  idx=nf-1 → (n-1, M+1) [or (M,M+1)]
    for idx in range(nf):
        j  = 2 + idx
        jm = n - 1 - idx
        A[:, 1, j ] = vals[:, idx]
        A[:, 1, jm] = vals[:, idx]

    # ── Rows 2 … n-1: Toeplitz shift of row 1 ─────────────────────────────────
    for i in range(2, n):
        if n - i - 1 > 0:
            A[:, i, i + 1:] = A[:, 1, 2 : n - i + 1]

    # ── Symmetrise lower triangle ──────────────────────────────────────────────
    iu = np.triu_indices(n, k=1)
    A[:, iu[1], iu[0]] = A[:, iu[0], iu[1]]

    return A % d


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Distribute canonical vectors evenly across MPI ranks
    per_rank = (TOTAL + size - 1) // size
    my_start = rank * per_rank
    my_end   = min(my_start + per_rank, TOTAL)
    my_n     = max(my_end - my_start, 0)

    if rank == 0:
        print(f"{'─'*62}")
        print(f"  AME({N},{D}) GPU search  [canonical form]")
        print(f"  Total canonical matrices : {TOTAL:>15,}")
        print(f"  MPI ranks                : {size:>15,}")
        print(f"  Matrices / rank          : {per_rank:>15,}")
        print(f"  GPUs visible             : {n_gpus:>15,}")
        print(f"  Batch size               : {BATCH:>15,}")
        print(f"  Bipartitions / matrix    : {NUM_SUBSETS:>15,}")
        print(f"  (Row 0 fixed = 1, 0∈K only, canonical row 1)")
        print(f"{'─'*62}", flush=True)

    # Pre-allocate GPU buffers (reused across batches)
    d_mats  = cuda.device_array((BATCH, N, N),        dtype=np.int32)
    d_flags = cuda.device_array((BATCH, NUM_SUBSETS), dtype=np.int8)
    d_mask  = cuda.device_array((BATCH,),             dtype=np.int8)

    TPB      = 256
    my_vals  = _all_vals[my_start:my_end]   # this rank's slice of canonical vectors
    ame_list = []

    for bs in range(0, my_n, BATCH):
        be   = min(bs + BATCH, my_n)
        nb   = be - bs                      # matrices in this (possibly partial) batch
        vals = my_vals[bs:be]

        # CPU: build batch of adjacency matrices
        h_mats = build_matrices(vals, N, D)

        # Upload to GPU
        d_mats[:nb].copy_to_device(h_mats)

        # Grid sizes for this batch
        bpg_k = (nb * NUM_SUBSETS + TPB - 1) // TPB
        bpg_r = (nb               + TPB - 1) // TPB

        # GPU: rank all bipartitions for every matrix
        ame_kernel[bpg_k, TPB](
            d_mats, d_K, d_C, d_flags, D, nb, NUM_SUBSETS, M
        )

        # GPU: AND-reduce → pass/fail per matrix
        reduce_kernel[bpg_r, TPB](d_flags, d_mask, nb, NUM_SUBSETS)

        cuda.synchronize()

        # Collect hits
        h_mask = d_mask[:nb].copy_to_host()
        hits   = np.where(h_mask)[0]
        if hits.size:
            for h in hits:
                ame_list.append(h_mats[h].copy())
            print(f"  [rank {rank:02d}] *** {hits.size} AME matrix(ces) found! ***",
                  flush=True)

        # Progress report (rank 0, every 10 batches)
        if rank == 0 and (bs // BATCH) % 10 == 0:
            pct = 100.0 * be / my_n
            print(f"  [rank 00] {be:>10,} / {my_n:,}  ({pct:6.2f}%)", flush=True)

    # Gather results at rank 0
    all_ame = comm.gather(ame_list, root=0)

    if rank == 0:
        combined = [m for sublist in all_ame for m in sublist]
        print(f"\n{'─'*62}")
        print(f"  DONE — found {len(combined)} AME({N},{D}) graph state(s).")
        if combined:
            out   = np.stack(combined)
            fname = f"ame_{N}_{D}_generators.npy"
            np.save(fname, out)
            print(f"  Saved {out.shape} array  →  {fname}")
        print(f"{'─'*62}")


if __name__ == "__main__":
    main()