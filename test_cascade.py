"""Structured-operand tests of the cascade bridge; the registered
fold-gain prediction lives here too. Helpers are in cascade.py."""

import numpy as np
from quantize import predicted_error, range_rule, apply_contraction
from solver import BETA, gen_geometric
from cascade import (cascade_field_rows, cascade_field_cols,
                     measured_sq_error, panel_step, wilkinson, nonneg)

rng = np.random.default_rng(1)

# ---- tests ----

def test_cascade_bridge_on_random_operands():
    A = rng.standard_normal((96, 64))
    B = rng.standard_normal((64, 96))
    ratios = []
    for d in (2, 3, 4):
        pred = predicted_error(A, B, cascade_field_rows(A, d), cascade_field_cols(B, d))
        meas = measured_sq_error(A, B, d)
        ratios.append(meas / pred)
        print(f"depth {d}: predicted {pred:.3e}  measured {meas:.3e}  ratio {meas/pred:.2f}")
    assert all(0.8 < r < 1.25 for r in ratios)


def test_identity_predicts_real_panel_operands():
    A = gen_geometric(256, 1e4, seed=3)
    L21, U12 = panel_step(A, 32)
    d = 3
    pred = predicted_error(L21, U12, cascade_field_rows(L21, d), cascade_field_cols(U12, d))
    meas = measured_sq_error(L21, U12, d)
    print(f"real L21/U12 panel, depth {d}: ratio measured/predicted = {meas/pred:.2f}")
    assert 0.7 < meas / pred < 1.4


def test_wrong_axis_field_mispredicts_on_structured_operands():
    # same panel, but describe U12 with a row-based field although it is
    # column-sliced; random operands would hide this, real ones do not
    A = gen_geometric(256, 1e4, seed=3)
    L21, U12 = panel_step(A, 32)
    d = 3
    right = predicted_error(L21, U12, cascade_field_rows(L21, d), cascade_field_cols(U12, d))
    wrong = predicted_error(L21, U12, cascade_field_rows(L21, d), cascade_field_rows(U12, d))
    meas = measured_sq_error(L21, U12, d)
    print(f"right-axis ratio {meas/right:.2f}, wrong-axis ratio {meas/wrong:.2f}")
    assert abs(np.log(meas / right)) < abs(np.log(meas / wrong))


def test_per_slice_factor_from_prediction():
    A = gen_geometric(256, 1e4, seed=3)
    L21, U12 = panel_step(A, 32)
    p = [predicted_error(L21, U12, cascade_field_rows(L21, d), cascade_field_cols(U12, d))
         for d in (2, 3, 4)]
    f1, f2 = np.sqrt(p[0] / p[1]), np.sqrt(p[1] / p[2])
    print(f"predicted per-slice factors {f1:.0f}, {f2:.0f}  (theory {BETA}, record 252)")
    assert 180 < f1 < 330 and 180 < f2 < 330


def test_registered_prediction_fold_gain_tracks_growth():
    # written before running: identity ~ optimal on benign spectra,
    # real gain where growth is nonuniform along the panel
    # re-registered 2026-09-03: the old single numbers (1.03/1.09/1.72)
    # were one draw each of a seed-sensitive experiment (wilkinson spans
    # roughly 1.4-3.7 at draws=8). the claim is now the median over three
    # jitter seeds at draws=20, and the asserts are bands, so a gross
    # shift trips a test instead of hiding in the prints
    cases = {
        "geometric k=1e4": gen_geometric(256, 1e4, seed=5),
        "nonneg random": nonneg(256),
        "wilkinson": wilkinson(256),
    }
    d, seeds = 3, (0, 1, 2)
    med = {}
    for name, A in cases.items():
        L21, U12 = panel_step(A, 32)
        Lf, Uf = apply_contraction(L21, U12, range_rule)
        g = []
        for sd in seeds:
            e_id = np.sqrt(measured_sq_error(L21, U12, d, draws=20, seed=sd))
            e_fold = np.sqrt(measured_sq_error(Lf, Uf, d, draws=20, seed=sd))
            g.append(e_id / e_fold)
        h = range_rule(L21, U12)
        med[name] = np.median(g)
        print(f"{name:16s} profile spread {np.max(h)/np.min(h):9.2e}"
              f"  fold gain median {med[name]:5.2f}x"
              f"  range [{min(g):.2f}, {max(g):.2f}]")
    assert 0.95 < med["geometric k=1e4"] < 1.2
    assert 0.95 < med["nonneg random"] < 1.35
    assert 1.5 < med["wilkinson"] < 3.5
    assert med["wilkinson"] > med["geometric k=1e4"]


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
