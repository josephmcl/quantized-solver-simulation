"""Tests for the reference dial solver.

Several of these encode measured facts from the paper (the staircase, the
column-equivariance bit-identity, the wall). If you change the quantizer
and a test here breaks, the paper's numbers are what you're contradicting.
"""

import numpy as np
import pytest
from solver import (BETA, INT_CAP, slice_rows, slice_cols,
                    reconstruct_rows, sliced_gemm, truncation_rule,
                    lu_int8, refine, equilibrate, kappa_eq, dial_depth,
                    solve, gen_geometric, wild_scale)

rng = np.random.default_rng(7)


def test_int_cap_constant():
    # the accumulator-safe contraction length; the paper prints 133,144
    print(f"int cap = {INT_CAP} (2^31 // 127^2)")
    assert INT_CAP == 133144


def test_slice_reconstruction_bound():
    A = rng.standard_normal((64, 64))
    for s in (1, 2, 4):
        err = np.abs(reconstruct_rows(slice_rows(A, s)) - A)
        rowmax = np.max(np.abs(A), axis=1)
        print(f"depth {s}: worst err/rowmax = {np.max(err / rowmax[:, None]):.2e}"
              f"  (bound {BETA**(-s):.2e})")
        assert np.all(err <= 1.01 * BETA**(-s) * rowmax[:, None])


def test_staircase_factor():
    # one more slice should buy about a factor of BETA; the record says 252
    A = rng.standard_normal((128, 128))
    e = []
    for s in (2, 3, 4):
        e.append(np.linalg.norm(reconstruct_rows(slice_rows(A, s)) - A))
    r1, r2 = e[0] / e[1], e[1] / e[2]
    print(f"per-slice factors: {r1:.0f}, {r2:.0f}  (theory {BETA}, record says 252)")
    assert 150 < r1 < 350 and 150 < r2 < 350


def test_gemm_exact_when_deep():
    A = rng.standard_normal((48, 32))
    B = rng.standard_normal((32, 40))
    C = sliced_gemm(slice_rows(A, 8), slice_cols(B, 8), lambda i, j: True)
    rel = np.linalg.norm(C - A @ B) / np.linalg.norm(A @ B)
    print(f"deep sliced gemm vs float64: rel err = {rel:.2e}")
    assert rel < 1e-13


def test_truncation_costs_a_constant_not_an_order():
    A = rng.standard_normal((48, 48))
    B = rng.standard_normal((48, 48))
    s = 4
    full = sliced_gemm(slice_rows(A, s), slice_cols(B, s), lambda i, j: True)
    kept = sliced_gemm(slice_rows(A, s), slice_cols(B, s), truncation_rule(s))
    ref = A @ B
    e_kept = np.linalg.norm(kept - ref)
    e_full = np.linalg.norm(full - ref)
    print(f"truncated/full error ratio = {e_kept / e_full:.1f}"
          "  (a constant, not a power of beta)")
    assert e_kept < 50 * e_full + 1e-12   # same order, bigger constant


def test_factor_gate_tracks_depth():
    A = gen_geometric(96, 1e4, seed=1)
    gates = [lu_int8(A, s, panel=32)[3] for s in (2, 3, 4)]
    print("gate ||PA-LU||/||A|| by depth: "
          + ", ".join(f"s={s}: {g:.2e}" for s, g in zip((2, 3, 4), gates)))
    assert gates[0] > gates[1] > gates[2]
    assert gates[1] / gates[2] > 50    # roughly a BETA per slice


def test_dial_law_values():
    # spot values, not sacred; the law is measured, the form is derived
    print(f"dial(8192, 1e6) = {dial_depth(8192, 1e6)},"
          f" dial(2048, 1e2) = {dial_depth(2048, 1e2)}")
    assert dial_depth(8192, 1e6) == 4
    assert dial_depth(2048, 1e2) <= 2


def test_solver_reaches_floor_in_range():
    n = 96
    A = gen_geometric(n, 1e4, seed=2)
    b = rng.standard_normal(n)
    out = solve(A, b)
    print(f"n={n} kappa=1e4: depth {out['depth']}, {out['iters']} iters,"
          f" bwd {out['bwd']:.2e}, gate {out['gate']:.2e}")
    assert out["bwd"] <= 1e-14
    assert out["iters"] <= 25


def test_wall_and_honest_flag():
    # two slices cannot carry kappa=1e8; the flag must say so
    n = 96
    A = gen_geometric(n, 1e8, seed=3)
    b = rng.standard_normal(n)
    L, U, piv, _ = lu_int8(A, 2, panel=32)
    x, its, bwd = refine(A, b, L, U, piv, maxit=30)
    print(f"kappa=1e8 at depth 2: stalled at bwd {bwd:.2e} after {its} sweeps")
    assert bwd > 1e-12        # did not reach the floor
    # one more slice pair changes the story
    L, U, piv, _ = lu_int8(A, 4, panel=32)
    x, its, bwd = refine(A, b, L, U, piv, maxit=40)
    print(f"same matrix at depth 4: {its} iters, bwd {bwd:.2e}")
    assert bwd <= 1e-13


def test_equilibration_mechanism():
    # wild scaling explodes kappa, not kappa_eq, and not the required depth
    A = gen_geometric(96, 1e3, seed=4)
    W, _, _ = wild_scale(A, spread=6, seed=4)
    print(f"wild-scaled: kappa = {np.linalg.cond(W):.1e},"
          f" kappa_eq = {kappa_eq(W):.1e},"
          f" dial says depth {dial_depth(96, kappa_eq(W))}")
    assert np.linalg.cond(W) > 1e8
    assert kappa_eq(W) < 1e5
    assert dial_depth(96, kappa_eq(W)) <= dial_depth(96, 1e3 * 1.5) + 1


def test_pow2_column_equivariance():
    # Proposition 3's machine check, miniature: exact pow2 column scaling
    # leaves pivots and L bit-identical and scales U exactly
    n, s = 64, 3
    A = gen_geometric(n, 1e4, seed=5)
    e = rng.integers(-20, 20, n)
    D = 2.0 ** e
    L0, U0, p0, _ = lu_int8(A, s, panel=16)
    L1, U1, p1, _ = lu_int8(A * D[None, :], s, panel=16)
    print(f"pow2 column scaling: pivots identical = {np.array_equal(p0, p1)},"
          f" L bit-identical = {np.array_equal(L0, L1)}")
    assert np.array_equal(p0, p1)
    assert np.array_equal(L0, L1)                    # bit for bit
    assert np.array_equal(U1, U0 * D[None, :])       # exact covariance


def test_row_scaling_is_not_covered():
    # the open half: row scaling may flip pivots, and often does
    n, s = 64, 3
    A = gen_geometric(n, 1e4, seed=6)
    e = rng.integers(-20, 20, n)
    _, _, p0, _ = lu_int8(A, s, panel=16)
    _, _, p1, _ = lu_int8((2.0 ** e)[:, None] * A, s, panel=16)
    print(f"row scaling flipped {np.sum(p0 != p1)} of {len(p0)} pivot entries")
    assert not np.array_equal(p0, p1)


def test_scale_invariant_signals_under_general_column_scaling():
    # general column scales can tip near-tie pivots; the scale-invariant
    # signals are depth and forward error at a fixed sweep count. normwise
    # backward error is deliberately not compared, it is distorted by the
    # column spread (see the campaign record's directional-split caveat)
    n, s, sweeps = 96, 3, 6
    A = gen_geometric(n, 1e4, seed=8)
    x_true = rng.standard_normal(n)
    _, _, d2 = wild_scale(A, spread=4, seed=8)
    W = A * d2[None, :]
    fwd = []
    for M, xt in ((A, x_true), (W, x_true / d2)):
        L, U, piv, _ = lu_int8(M, s, panel=32)
        x = np.zeros(n)
        for _ in range(sweeps):
            r = M @ xt - M @ x
            x = x + np.linalg.solve(U, np.linalg.solve(
                np.tril(L, -1) + np.eye(n), r[piv]))
        fwd.append(np.linalg.norm(x - xt) / np.linalg.norm(xt))
    print(f"fwd err after {sweeps} sweeps: base {fwd[0]:.2e},"
          f" column-scaled {fwd[1]:.2e}")
    assert fwd[1] < 5 * fwd[0] + 1e-15 and fwd[0] < 5 * fwd[1] + 1e-15


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))