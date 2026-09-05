"""Seeded quotient-domain stochastic rounding for the slicer, the Tier-2
design option of the grounding program. Dither lives on the scale-free
quotient with a counter-based hash keyed on indices, so determinism,
cross-backend reproducibility, and pow2 column equivariance all survive;
the price is a smaller per-slice base, measured in the tests.

R1 (reconciliation, 2026-09-05). ONE dither spec, ONE canonical
signature. The key of a rounding decision is the tuple

    (position i, position j, level, step, stream salt)

and every rounding decision in this simulation is a pure function of it:
no RNG state, no call order, no data dependence. `step` is the ancestor
field that REPORT.md's T2 disagreement proved mandatory -- it must
advance with the update / panel index, or the frozen field's (1/2 - u)
conditional bias accumulates coherently across steps. It is part of the
spec, not an option; the old `key=` argument (a single stream, frozen at
its default) is gone, and with it the frozen-key-only spec.

Schedules that obey or violate the spec on purpose are built in
keyed_sr.py out of this one function:

    S1  global-fresh         a fresh `salt` per rounding (execution order)
    S2  canonical            step = update index              <- the spec
    S3  ancestor-violating   step held fixed (the frozen field)
    S3b parallel-violating   share_j=True: one draw shared across the
                             b contraction positions of an output

Rounding mode of everything here is SR; RTN lives in solver.slice_rows.
"""

import numpy as np

QMAX = 127

# stream salts: the row operand and the column operand of a product are
# distinct streams, so a shared position never means a shared draw
SALT_ROWS = 0x9E3779B97F4A7C15
SALT_COLS = 0x51ED270100000001

_MASK64 = (1 << 64) - 1
_M_I = np.uint64(0x9E3779B97F4A7C15)
_M_J = np.uint64(0xC2B2AE3D27D4EB4F)


def _mix64(h):
    # splitmix64 finalizer, in python ints so uint64 scalar overflow
    # (which numpy warns about) never happens; arrays wrap silently
    h &= _MASK64
    h ^= h >> 30
    h = (h * 0xBF58476D1CE4E5B9) & _MASK64
    h ^= h >> 27
    h = (h * 0x94D049BB133111EB) & _MASK64
    return h ^ (h >> 31)


def _mix_arr(h):
    h ^= h >> np.uint64(30)
    h = h * np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h = h * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def base_key(level, step=0, salt=SALT_ROWS):
    """The non-positional half of the key. Separated so keyed_sr can name
    a schedule by what it does to (level, step, salt)."""
    return _mix64(salt + _mix64(step * 0x27D4EB2F165667C5
                                + _mix64(level * 0x165667B19E3779F9
                                         + 0x94D049BB133111EB)))


def dither(shape, level, step=0, salt=SALT_ROWS, share_j=False):
    """u in [0, 1) for every position of `shape`, keyed on
    (i, j, level, step, salt). Counter-based: a pure function of the key,
    so the field is order-free, backend-free and scale-free.

    share_j drops the second index from the key, so one draw is shared
    across a whole row -- across the b contraction positions, when the
    operand is laid out as a slicing calls it. That is the parallel-
    channel violation (keyed_sr S3b); the canonical spec never sets it.
    """
    h = np.uint64(base_key(level, step, salt))
    i = np.arange(shape[0], dtype=np.uint64)[:, None]
    j = (np.zeros(shape[1], dtype=np.uint64) if share_j
         else np.arange(shape[1], dtype=np.uint64))[None, :]
    u = _mix_arr(h + i * _M_I + j * _M_J)
    return (u >> np.uint64(11)).astype(np.float64) / np.float64(1 << 53)


def sr_slice_rows(A, depth, step=0, salt=SALT_ROWS, share_j=False):
    # per-entry error is mean zero with variance frac(1-frac) quanta^2,
    # which makes the variance-field identity exact rather than approximate.
    # callers doing repeated updates MUST advance `step`; see the module
    # docstring and test_accumulation.py.
    R = A.astype(np.float64).copy()
    out = []
    for lvl in range(depth):
        m = np.max(np.abs(R), axis=1)
        s = np.where(m > 0, m / QMAX, 1.0)
        q = R / s[:, None]
        f = np.floor(q)
        u = dither(A.shape, lvl, step, salt, share_j)
        d = (f + (u < (q - f))).astype(np.int8)
        out.append((d, s))
        R = R - d.astype(np.float64) * s[:, None]
    return out


def sr_slice_cols(B, depth, step=0, salt=SALT_COLS, share_j=False):
    # B is sliced along its columns, i.e. rows of B.T; the contraction
    # index of B is then the *second* index of B.T, so share_j means the
    # same thing on both operands of a product
    return [(d.T, s) for d, s in
            sr_slice_rows(np.ascontiguousarray(B.T), depth, step, salt, share_j)]
