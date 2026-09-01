"""Checks for quantize.py

The predicted-error identity is only a formula until it is measured against
real rounding; the first test does that. The second shows the fold machinery
does something on an operand pair built to need it.
"""

import numpy as np
from quantize import (quantize_by_row, variance, predicted_error,
                      range_rule, apply_contraction)

rng = np.random.default_rng(0)
b = 8


def measured_sq_error(A, B, draws=30):
    # jitter the operands so each draw samples the rounding lattice afresh,
    # deterministic RTN on identical inputs would just repeat one number
    out = []
    for _ in range(draws):
        Aj = A * (1 + 1e-6 * rng.standard_normal(A.shape))
        Bj = B * (1 + 1e-6 * rng.standard_normal(B.shape))
        Ah, Bh = quantize_by_row(Aj, b), quantize_by_row(Bj, b)
        out.append(np.linalg.norm(Ah @ Bh - Aj @ Bj) ** 2)
    return np.mean(out)


def test_identity_matches_measurement():
    # Thm 3.3 against real round-to-nearest, benign operands
    A = rng.standard_normal((96, 96))
    B = rng.standard_normal((96, 96))
    pred = predicted_error(A, B, variance(A, b), variance(B, b))
    meas = measured_sq_error(A, B)
    print(f"predicted {pred:.3e}  measured {meas:.3e}  ratio {meas / pred:.2f}")
    assert 0.85 < meas / pred < 1.15


def test_range_rule_fold_helps_when_needed():
    # anti-correlated contraction energies, the case folds exist for
    A = rng.standard_normal((96, 96))
    B = rng.standard_normal((96, 96))
    scale = np.logspace(-3, 3, 96)
    A2, B2 = A * scale[None, :], B / scale[:, None]
    Ah, Bh = quantize_by_row(A2, b), quantize_by_row(B2, b)
    e_id = np.linalg.norm(Ah @ Bh - A2 @ B2)
    Af, Bf = apply_contraction(A2, B2, range_rule)
    Ah, Bh = quantize_by_row(Af, b), quantize_by_row(Bf, b)
    e_fold = np.linalg.norm(Ah @ Bh - Af @ Bf)
    print(f"identity {e_id:.3e}  folded {e_fold:.3e}  gain {e_id / e_fold:.0f}x")
    assert e_fold < e_id / 10


def test_fold_is_a_gauge():
    # (AT)(T^-1 B) must be the same product before any quantization
    A = rng.standard_normal((40, 40))
    B = rng.standard_normal((40, 40))
    Af, Bf = apply_contraction(A, B, range_rule)
    assert np.allclose(Af @ Bf, A @ B, rtol=1e-12)


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))