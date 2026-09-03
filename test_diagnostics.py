"""Checks for the lattice diagnostic (Xi prototype)."""

import numpy as np
from quantize import predicted_error
from solver import slice_rows, reconstruct_rows
from cascade import (cascade_field_rows, cascade_field_cols,
                     measured_sq_error, panel_step, wilkinson)
from diagnostics import xi

rng = np.random.default_rng(2)


def test_lattice_diagnostic_separates_regimes():
    # registered prediction (written before running, RTN): the diagnostic
    # value and the identity-ratio deviation move together, and the benign
    # to adversarial gap in KS distance is at least 5x. benign random
    # operands sit at identity ratio 0.9-1.0 as recorded; an integer-valued
    # operand pair (few distinct values, lattice quotients) trips the
    # diagnostic AND shows the larger ratio deviation.
    A = rng.standard_normal((128, 128))
    B = rng.standard_normal((128, 128))
    Ai = rng.integers(-5, 6, (128, 128)).astype(float)
    Bi = rng.integers(-5, 6, (128, 128)).astype(float)
    d = 3
    out = {}
    for name, (X, Y) in {"benign": (A, B), "integer": (Ai, Bi)}.items():
        pred = predicted_error(X, Y, cascade_field_rows(X, d),
                               cascade_field_cols(Y, d))
        meas = measured_sq_error(X, Y, d, seed=0)
        mdev, ks = xi(X)
        out[name] = (ks, meas / pred)
        print(f"{name:8s} xi: mean dev {mdev:.4f}, KS {ks:.4f}"
              f"   identity ratio {meas/pred:.3f}")
    assert 0.85 < out["benign"][1] < 1.05
    assert out["integer"][0] > 5 * out["benign"][0]
    assert abs(np.log(out["integer"][1])) > abs(np.log(out["benign"][1]))


def test_lattice_extreme_case():
    # extreme case, print-focused per the brief: wilkinson's L21 is all
    # +-1, one exact slice, so the error is zero and the ratio degenerate
    # rather than degraded. the diagnostic saturates (all fracs 0, KS 1)
    L21, _ = panel_step(wilkinson(256), 32)
    mdev, ks = xi(L21)
    exact = np.linalg.norm(reconstruct_rows(slice_rows(L21, 1)) - L21)
    print(f"wilkinson L21 xi: mean dev {mdev:.4f}, KS {ks:.4f}"
          f"   one-slice reconstruction error {exact:.1e} (exact)")
    assert ks > 0.9
    assert exact == 0.0
