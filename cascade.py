"""The bridge between the layered quantizer and the variance-field model,
plus the panel extraction and matrix families the structured tests need.
"""

import numpy as np
from solver import slice_rows, slice_cols, sliced_gemm, truncation_rule
from quantize import predicted_error

ALL = lambda i, j: True


def cascade_field_rows(A, depth):
    # after depth slices the leftover error per entry is uniform on
    # +-(last scale)/2, so the field is (last scale)^2/12 per row
    s_last = slice_rows(A, depth)[-1][1]
    return np.broadcast_to((s_last ** 2 / 12)[:, None], A.shape).copy()

def cascade_field_cols(B, depth):
    t_last = slice_cols(B, depth)[-1][1]
    return np.broadcast_to((t_last ** 2 / 12)[None, :], B.shape).copy()

def measured_sq_error(A, B, depth, draws=20, keep=ALL, seed=0):
    # hermetic on purpose; a shared module stream made results depend on
    # call order, which is how the 1.72x drift happened
    g = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        Aj = A * (1 + 1e-7 * g.standard_normal(A.shape))
        Bj = B * (1 + 1e-7 * g.standard_normal(B.shape))
        C = sliced_gemm(slice_rows(Aj, depth), slice_cols(Bj, depth), keep)
        out.append(np.linalg.norm(C - Aj @ Bj) ** 2)
    return np.mean(out)


def omitted_pairs_error(A, B, depth):
    # exact Frobenius norm of the slice pairs the shipped rule drops;
    # deterministic under RTN, no draws involved
    Asl = slice_rows(A, depth)
    Bsl = slice_cols(B, depth)
    full = sliced_gemm(Asl, Bsl, ALL)
    kept = sliced_gemm(Asl, Bsl, truncation_rule(depth))
    return np.linalg.norm(full - kept)


def panel_step(A, kb):
    # one blocked-LU panel step, returns the real L21/U12 trailing operands
    W = A.astype(np.float64).copy()
    for c in range(kb):
        p = c + np.argmax(np.abs(W[c:, c]))
        W[[c, p], :] = W[[p, c], :]
        W[c+1:, c] /= W[c, c]
        W[c+1:, c+1:kb] -= np.outer(W[c+1:, c], W[c, c+1:kb])
    L11 = np.tril(W[:kb, :kb], -1) + np.eye(kb)
    return W[kb:, :kb], np.linalg.solve(L11, W[:kb, kb:])


def wilkinson(n):
    # the classical growth-2^(n-1) matrix under partial pivoting
    W = -np.tril(np.ones((n, n)), -1) + np.eye(n)
    W[:, -1] = 1.0
    return W

def nonneg(n, seed=1):
    return np.random.default_rng(seed).uniform(0.0, 1.0, (n, n))