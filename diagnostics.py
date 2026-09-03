"""Lattice diagnostic, the Xi prototype for the certificate loop: a cheap
per-operand check of whether RTN is in the regime the stochastic identity
covers (M2, uniform final-slice residuals). Works on the quotient fracs of
row slicing at a chosen cascade level; benign operands give near-uniform
fracs, lattice-structured operands concentrate them.
"""

import numpy as np
from solver import slice_rows, reconstruct_rows

QMAX = 127


def frac_field(A, level=0):
    # fracs f = q - floor(q) of the row-slicing quotients at the given
    # cascade level (level 0 is the operand itself)
    R = A.astype(np.float64)
    if level > 0:
        R = R - reconstruct_rows(slice_rows(A, level))
    m = np.max(np.abs(R), axis=1)
    s = np.where(m > 0, m / QMAX, 1.0)
    q = R / s[:, None]
    return q - np.floor(q)


def xi(A, level=0):
    # the diagnostic pair: (|mean(f) - 0.5|, KS distance of f from uniform)
    f = np.sort(frac_field(A, level).ravel())
    n = f.size
    hi = np.arange(1, n + 1) / n
    ks = np.max(np.maximum(hi - f, f - (hi - 1.0 / n)))
    return abs(f.mean() - 0.5), ks
