"""Scale-controlled accumulation experiment (P-B). The gate-versus-n sweep
measures a composite of accumulation and the family's scale profile; this
isolates accumulation by feeding iid fresh standard-normal operands at
every step, so the per-step error scale is held flat by construction.
"""

import numpy as np
from solver import slice_rows, slice_cols, sliced_gemm, truncation_rule


def error_growth(m=192, b=32, depth=3, k=128, seed=0, slicers=None,
                 keep=None, detail=False):
    # k successive trailing updates, fresh L (m x b) and U (b x m) each
    # step; the sliced update's error accumulates into E. returns
    # ||E||_F after each step. slicers(step) -> (row_slicer, col_slicer)
    # so SR callers can vary the dither key per step; default is RTN.
    # keep overrides the shipped truncation rule (E2 uses ALL to isolate
    # the truncation channel). detail additionally returns the per-step
    # error norms and the final accumulated error field.
    if slicers is None:
        slicers = lambda t: (slice_rows, slice_cols)
    if keep is None:
        keep = truncation_rule(depth)
    g = np.random.default_rng(seed)
    E = np.zeros((m, m))
    cum, per = [], []
    for t in range(k):
        rows, cols = slicers(t)
        L = g.standard_normal((m, b))
        U = g.standard_normal((b, m))
        X = sliced_gemm(rows(L, depth), cols(U, depth), keep) - L @ U
        E += X
        cum.append(np.linalg.norm(E))
        per.append(np.linalg.norm(X))
    if detail:
        return np.array(cum), np.array(per), E
    return np.array(cum)


def fit_slope(errs, k_min=16):
    # log-log slope of ||E||_F versus step count over steps >= k_min
    ks = np.arange(1, len(errs) + 1)
    m = ks >= k_min
    return np.polyfit(np.log(ks[m]), np.log(errs[m]), 1)[0]
