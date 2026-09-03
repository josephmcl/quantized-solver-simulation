"""Seeded quotient-domain stochastic rounding for the slicer, the Tier-2
design option of the grounding program. Dither lives on the scale-free
quotient with a counter-based hash keyed on indices, so determinism,
cross-backend reproducibility, and pow2 column equivariance all survive;
the price is a smaller per-slice base, measured in the tests.
"""

import numpy as np

QMAX = 127


def dither(shape, level, key=0x9E3779B9):
    # counter-based, order-independent, part of the kernel spec
    i = np.arange(shape[0], dtype=np.uint64)[:, None]
    j = np.arange(shape[1], dtype=np.uint64)[None, :]
    h = (i * np.uint64(2654435761) ^ (j * np.uint64(40503))
         ^ np.uint64(level * 97 + key)) * np.uint64(2246822519)
    h ^= h >> np.uint64(13)
    return ((h * np.uint64(2654435761)) >> np.uint64(40)).astype(np.float64) / 2**24


def sr_slice_rows(A, depth, key=0x9E3779B9):
    # per-entry error is mean zero with variance frac(1-frac) quanta^2,
    # which makes the variance-field identity exact rather than approximate.
    # key selects the dither stream; callers doing repeated updates must
    # vary it per step or the fixed field's (0.5 - u) bias accumulates
    # coherently (see test_accumulation.py)
    R = A.astype(np.float64).copy()
    out = []
    for lvl in range(depth):
        m = np.max(np.abs(R), axis=1)
        s = np.where(m > 0, m / QMAX, 1.0)
        q = R / s[:, None]
        f = np.floor(q)
        d = (f + (dither(A.shape, lvl, key) < (q - f))).astype(np.int8)
        out.append((d, s))
        R = R - d.astype(np.float64) * s[:, None]
    return out


def sr_slice_cols(B, depth, key=0x9E3779B9):
    return [(d.T, s)
            for d, s in sr_slice_rows(np.ascontiguousarray(B.T), depth, key)]
