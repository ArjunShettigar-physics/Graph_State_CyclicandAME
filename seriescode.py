import numpy as np
from itertools import combinations, product

N = 18
D = 5

M        = N // 2
NUM_FREE = M - 1


def generate_canonical_rows(nf, p):
    rows = []
    for k in range(nf):
        n_tail = nf - k - 1
        for tail in product(range(p), repeat=n_tail):
            rows.append([0] * k + [1] + list(tail))
    return np.array(rows, dtype=np.int32)


def make_bipartitions(n, m):
    K_list = [k for k in combinations(range(n), m) if 0 in k]
    K = np.array(K_list, dtype=np.int32)
    C = np.zeros_like(K)
    for s, row in enumerate(K_list):
        C[s] = sorted(set(range(n)) - set(row))
    return K, C


def build_matrix(vals, n, d):
    nf = len(vals)
    A  = np.zeros((n, n), dtype=np.int32)
    A[0, 1:] = 1
    for idx in range(nf):
        A[1, 2 + idx]     = vals[idx]
        A[1, n - 1 - idx] = vals[idx]
    for i in range(2, n):
        if n - i - 1 > 0:
            A[i, i + 1:] = A[1, 2 : n - i + 1]
    iu = np.triu_indices(n, k=1)
    A[iu[1], iu[0]] = A[iu[0], iu[1]]
    return A % d


def gfp_rank(mat, m, p):
    L  = mat.astype(np.int64).copy()
    pp = p * p
    rk = 0
    for c in range(m):
        piv = -1
        for row in range(rk, m):
            if L[row, c] != 0:
                piv = row
                break
        if piv < 0:
            continue
        if piv != rk:
            L[[rk, piv]] = L[[piv, rk]]
        inv = pow(int(L[rk, c]), p - 2, p)
        L[rk] = (L[rk] * inv) % p
        for row in range(m):
            if row != rk and L[row, c] != 0:
                fac = int(L[row, c])
                L[row] = (L[row] - fac * L[rk] + pp) % p
        rk += 1
        if rk == m:
            break
    return rk


def is_ame(mat, K, C, m, p):
    for s in range(len(K)):
        if gfp_rank(mat[np.ix_(K[s], C[s])], m, p) < m:
            return False
    return True


def main():
    all_vals = generate_canonical_rows(NUM_FREE, D)
    K, C     = make_bipartitions(N, M)
    ame_list = []

    for vals in all_vals:
        mat = build_matrix(vals, N, D)
        if is_ame(mat, K, C, M, D):
            ame_list.append(mat.copy())

    print(f"Found {len(ame_list)} AME({N},{D}) graph state(s).")

    if ame_list:
        out = np.stack(ame_list)
        np.save(f"ame_{N}_{D}.npy", out)


if __name__ == "__main__":
    main()
