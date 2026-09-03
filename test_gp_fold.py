"""Toy-level checks of the certified fold optimizer; our panels are the
open experiment and deliberately not asserted here."""

import numpy as np
from quantize import range_rule
from gp_fold import solve_fold, exact_objective, identity_lp_test

rng = np.random.default_rng(0)


def test_gp_at_least_matches_range_rule_on_toy():
    A = rng.standard_normal((96, 96)); B = rng.standard_normal((96, 96))
    s = np.logspace(-3, 3, 96)
    A2, B2 = A * s[None, :], B / s[:, None]
    h, gain_gp = solve_fold(A2, B2)
    y = np.log(range_rule(A2, B2)); y -= y.mean()
    gain_rr = exact_objective(np.zeros(96), A2, B2) / exact_objective(y, A2, B2)
    print(f"predicted-error gain: range rule {gain_rr:.3e}, GP {gain_gp:.3e}")
    assert gain_gp >= 0.99 * gain_rr


def test_identity_not_stationary_on_generic_profile():
    A = rng.standard_normal((48, 48)); B = rng.standard_normal((48, 48))
    s = np.logspace(-2, 2, 48)
    feas, _ = identity_lp_test(A * s[None, :], B / s[:, None])
    print(f"identity stationary on imbalanced toy: {feas}")
    assert not feas