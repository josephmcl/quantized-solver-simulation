"""
NOTE: This currently implemented using FP64 where DF32 would otherwise be used, 
      so the floor is u_64, not u_32^2. Slicing mechanisms shouls otherwise 
      behave the same. 
"""

import numpy as np

BETA = 254            # per-slice error reduction, = 2*127 (symmetric grid, half-step rounding)
QMAX = 127
INT_CAP = 2**31 // (QMAX * QMAX)   # 133144, the accumulator-safe contraction length


# ---------------- slicing ----------------

def slice_rows(A, depth):
    # per-row max scaling, cascade on the residual
    # returns list of (int8 digits, per-row scales)
    R = A.astype(np.float64).copy()
    out = []
    for _ in range(depth):
        m = np.max(np.abs(R), axis=1)
        s = np.where(m > 0, m / QMAX, 1.0)
        d = np.rint(R / s[:, None]).astype(np.int8)   # ties-to-even, same as the kernels
        out.append((d, s))
        R = R - d.astype(np.float64) * s[:, None]   # stored-scale update, matters
    return out

def slice_cols(A, depth):
    sl = slice_rows(np.ascontiguousarray(A.T), depth)
    return [(d.T, s) for d, s in sl]

def reconstruct_rows(slices):
    A = np.zeros((slices[0][0].shape), dtype=np.float64)
    for d, s in slices:
        A += d.astype(np.float64) * s[:, None]
    return A


def sliced_gemm(Asl, Bsl, keep):
    """C = A @ B from row slices of A and column slices of B.

    keep(i, j) says which slice pairs survive. Integer products accumulate
    in int64 (int32 would do up to K=INT_CAP, but numpy makes int64 free).
    """
    da0, _ = Asl[0]
    _, sb0 = Bsl[0]
    C = np.zeros((da0.shape[0], Bsl[0][0].shape[1]))
    for i, (da, sa) in enumerate(Asl):
        for j, (db, sb) in enumerate(Bsl):
            if not keep(i, j):
                continue
            P = da.astype(np.int64) @ db.astype(np.int64)   # exact
            C += sa[:, None] * P.astype(np.float64) * sb[None, :]
    return C


def truncation_rule(depth):
    # keep pairs i+j < depth, the rule the real solver ships;
    # the dropped tail lands at order beta^-depth and inflates the constant
    return lambda i, j: i + j < depth


# ---------------- factorization ----------------

def lu_int8(A, depth, panel=64):
    """Right-looking blocked LU, partial pivoting, sliced trailing updates.

    Panels and triangular solves run in the float64 stand-in. Returns
    L, U, piv and the factor residual norm ||PA - LU||_F / ||A||_F, which
    is the gate quantity (H(p) certificate in the paper).
    """
    n = A.shape[0]
    assert panel <= INT_CAP
    W = A.astype(np.float64).copy()
    piv = np.arange(n)

    for k in range(0, n, panel):
        kb = min(panel, n - k)
        # panel, unblocked with row swaps applied across the full width
        for c in range(k, k + kb):
            p = c + np.argmax(np.abs(W[c:, c]))   # argmax is column-scale invariant, see Prop 3
            if p != c:
                W[[c, p], :] = W[[p, c], :]
                piv[[c, p]] = piv[[p, c]]
            W[c+1:, c] /= W[c, c]
            end = k + kb
            W[c+1:, c+1:end] -= np.outer(W[c+1:, c], W[c, c+1:end])
        if k + kb < n:
            # U12 by triangular solve against unit L11
            L11 = np.tril(W[k:k+kb, k:k+kb], -1) + np.eye(kb)
            W[k:k+kb, k+kb:] = np.linalg.solve(L11, W[k:k+kb, k+kb:])
            # trailing update through the quantizer
            L21 = W[k+kb:, k:k+kb]
            U12 = W[k:k+kb, k+kb:]
            upd = sliced_gemm(slice_rows(L21, depth), slice_cols(U12, depth),
                              truncation_rule(depth))
            W[k+kb:, k+kb:] -= upd

    L = np.tril(W, -1) + np.eye(n)
    U = np.triu(W)
    res = np.linalg.norm(A[piv] - L @ U) / np.linalg.norm(A)
    return L, U, piv, res


# ---------------- solve ----------------

def refine(A, b, L, U, piv, tol=1e-14, maxit=60):
    """LU-IR against the true operator. Residual in the float64 stand-in.

    Returns x, iteration count, final normwise backward error. Converged
    means bwd <= tol; the flag can't lie because the residual uses A itself.
    """
    x = np.zeros_like(b, dtype=np.float64)
    nrmA = np.linalg.norm(A, np.inf)
    for it in range(1, maxit + 1):
        r = b - A @ x
        bwd = np.linalg.norm(r, np.inf) / (nrmA * np.linalg.norm(x, np.inf)
                                           + np.linalg.norm(b, np.inf))
        if bwd <= tol:
            return x, it - 1, bwd   # residual uses A itself, so this flag is honest
        d = np.linalg.solve(U, np.linalg.solve(
            np.tril(L, -1) + np.eye(len(b)), r[piv]))
        x = x + d
    return x, maxit, bwd


# ---------------- the dial ----------------

def equilibrate(A, iters=10):
    # two-sided scaling to unit row and column maxima, Ruiz's iteration
    # (RAL-TR-2001-034); sqrt so the two sides take half steps and converge
    d1 = np.ones(A.shape[0]); d2 = np.ones(A.shape[1])
    B = A.copy().astype(np.float64)
    for _ in range(iters):
        r = np.sqrt(np.max(np.abs(B), axis=1))
        c = np.sqrt(np.max(np.abs(B), axis=0))
        r[r == 0] = 1; c[c == 0] = 1
        B = B / r[:, None] / c[None, :]
        d1 *= r; d2 *= c
    return B, d1, d2

def kappa_eq(A):
    B, _, _ = equilibrate(A)
    return np.linalg.cond(B)

C_DIAL = 0.09

def dial_depth(n, keq):
    # sigma score, then the ceiling; policy delta_s handled by the caller
    # real-valued score; the integer depth is its ceiling
    sigma = np.log(C_DIAL * n**0.85 * keq) / np.log(BETA)
    return max(1, int(np.ceil(sigma)))


def solve(A, b, delta_s=0, tol=1e-14):
    """Factor at the dial's depth, refine to the floor. The whole method."""
    n = A.shape[0]
    s = dial_depth(n, kappa_eq(A)) + delta_s
    L, U, piv, gate = lu_int8(A, s)
    x, its, bwd = refine(A, b, L, U, piv, tol=tol)
    return {"x": x, "depth": s, "iters": its, "bwd": bwd, "gate": gate}


# ---------------- generators ----------------

def gen_geometric(n, kappa, seed=0):
    # randsvd with geometric spectrum, the workhorse test family
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sv = kappa ** (-np.arange(n) / (n - 1))
    return U * sv @ V.T

def wild_scale(A, spread, seed=0, pow2=False):
    # D1 A D2 with log-uniform (or exact power of two) diagonals
    rng = np.random.default_rng(seed)
    e1 = rng.uniform(-spread, spread, A.shape[0])
    e2 = rng.uniform(-spread, spread, A.shape[1])
    if pow2:
        d1 = 2.0 ** np.rint(e1 * np.log2(10))
        d2 = 2.0 ** np.rint(e2 * np.log2(10))
    else:
        d1 = 10.0 ** e1
        d2 = 10.0 ** e2
    return d1[:, None] * A * d2[None, :], d1, d2