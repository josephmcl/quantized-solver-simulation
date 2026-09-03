"""Checks for the stochastic-rounding slicer, each a claim from the
grounding program with its measured number."""

import numpy as np
import solver
from solver import gen_geometric, lu_int8
from quantize import predicted_error
from sr_slicing import sr_slice_rows, dither, QMAX

rng = np.random.default_rng(3)


def test_pow2_column_equivariance_survives():
    A = rng.standard_normal((64, 64))
    e = rng.integers(-15, 15, 64)
    S0 = sr_slice_rows(A.T, 3)
    S1 = sr_slice_rows((A * (2.0 ** e)[None, :]).T, 3)
    ok = all(np.array_equal(d0, d1) for (d0, _), (d1, _) in zip(S0, S1))
    print(f"SR pow2 equivariance, digits bit-identical: {ok}")
    assert ok


def test_per_slice_base():
    A = rng.standard_normal((128, 128))
    errs = []
    for s in (1, 2, 3, 4):
        rec = sum(d.astype(float) * sc[:, None] for d, sc in sr_slice_rows(A, s))
        errs.append(np.linalg.norm(rec - A))
    ratios = [errs[i] / errs[i + 1] for i in range(3)]
    print("SR per-slice factors: " + ", ".join(f"{r:.0f}" for r in ratios)
          + "  (RTN ~254; worst case said 127; cost ~ half a slice)")
    assert all(115 < r < 185 for r in ratios)


def test_identity_exact_under_sr():
    A = rng.standard_normal((80, 80)); B = rng.standard_normal((80, 80))
    sA = np.max(np.abs(A), axis=1) / QMAX
    sB = np.max(np.abs(B), axis=0) / QMAX
    qA = A / sA[:, None]; fA = qA - np.floor(qA)
    qB = B / sB[None, :]; fB = qB - np.floor(qB)
    vA = fA * (1 - fA) * sA[:, None] ** 2
    vB = fB * (1 - fB) * sB[None, :] ** 2
    pred = predicted_error(A, B, vA, vB)
    meas = []
    for t in range(200):
        dA = np.floor(qA) + (dither(A.shape, 0, key=t) < fA)
        dB = np.floor(qB.T) + (dither(B.T.shape, 1, key=t) < fB.T)
        meas.append(np.linalg.norm((dA * sA[:, None]) @ (dB.T * sB[None, :]) - A @ B) ** 2)
    r = np.mean(meas) / pred
    print(f"SR identity exactness: measured/predicted = {r:.3f}")
    assert 0.95 < r < 1.05


def test_composite_slope_same_under_both_roundings():
    # the finding: the sub-worst-case gate exponent is scale-profile driven,
    # not rounding-statistics driven; RTN and SR must agree
    b, depth = 32, 3
    ns = [64, 128, 256, 512]
    def sweep():
        g = [lu_int8(gen_geometric(n, 1e4, seed=n), depth, panel=b)[3] for n in ns]
        return np.polyfit(np.log([n / b for n in ns]), np.log(g), 1)[0]
    s_rtn = sweep()
    saved = solver.slice_rows
    solver.slice_rows = sr_slice_rows
    try:
        s_sr = sweep()
    finally:
        solver.slice_rows = saved
    print(f"composite gate slope: RTN {s_rtn:.2f}, SR {s_sr:.2f}"
          "  (worst case 1.0, independence alone 0.5)")
    assert abs(s_rtn - s_sr) < 0.1
    assert 0.5 < s_rtn < 1.0
