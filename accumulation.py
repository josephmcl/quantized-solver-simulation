"""Scale-controlled accumulation experiment (P-B). The gate-versus-n sweep
measures a composite of accumulation and the family's scale profile; this
isolates accumulation by feeding iid fresh standard-normal operands at
every step, so the per-step error scale is held flat by construction.
"""

import numpy as np
from solver import slice_rows, slice_cols, sliced_gemm, truncation_rule


def error_growth(m=192, b=32, depth=3, k=128, seed=0, slicers=None):
    # k successive trailing updates, fresh L (m x b) and U (b x m) each
    # step; the sliced update's error accumulates into E. returns
    # ||E||_F after each step. slicers(step) -> (row_slicer, col_slicer)
    # so SR callers can vary the dither key per step; default is RTN.
    if slicers is None:
        slicers = lambda t: (slice_rows, slice_cols)
    g = np.random.default_rng(seed)
    keep = truncation_rule(depth)
    E = np.zeros((m, m))
    out = []
    for t in range(k):
        rows, cols = slicers(t)
        L = g.standard_normal((m, b))
        U = g.standard_normal((b, m))
        upd = sliced_gemm(rows(L, depth), cols(U, depth), keep)
        E += upd - L @ U
        out.append(np.linalg.norm(E))
    return np.array(out)


def fit_slope(errs, k_min=16):
    # log-log slope of ||E||_F versus step count over steps >= k_min
    ks = np.arange(1, len(errs) + 1)
    m = ks >= k_min
    return np.polyfit(np.log(ks[m]), np.log(errs[m]), 1)[0]
