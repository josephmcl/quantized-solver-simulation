"""Structured-operand tests of the cascade bridge; the registered
fold-gain prediction lives here too. Helpers are in cascade.py."""

import numpy as np
from quantize import predicted_error, range_rule, apply_contraction
from solver import BETA, gen_geometric
from cascade import (cascade_field_rows, cascade_field_cols,
                     measured_sq_error, omitted_pairs_error, panel_step,
                     wilkinson, nonneg)

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
    # registered before running: identity ~ optimal on benign spectra, real
    # gain where growth is nonuniform along the panel; gain ordered by the
    # contraction profile spread. findings are seed-dependent, so the claim
    # is a median plus range OVER A PRINTED SEED SET, banded from the
    # cross-platform union of observed draws (wilkinson has been seen from
    # 1.45x to 2.86x across machines; the volatility is itself a property
    # of extreme-growth panels). never quote the median without the range.
    d = 3
    seeds = (5, 11, 17, 23, 31)
    med, spr = {}, {}
    for name, gen in {
        "geometric k=1e4": lambda sd: gen_geometric(256, 1e4, seed=sd),
        "nonneg random": lambda sd: nonneg(256, seed=sd),
        "wilkinson": lambda sd: wilkinson(256),   # matrix fixed, jitter varies
    }.items():
        gains, spreads = [], []
        for sd in seeds:
            L21, U12 = panel_step(gen(sd), 32)
            h = range_rule(L21, U12)
            spreads.append(np.max(h) / np.min(h))
            e_id = np.sqrt(measured_sq_error(L21, U12, d, draws=8, seed=sd))
            Lf, Uf = apply_contraction(L21, U12, range_rule)
            e_f = np.sqrt(measured_sq_error(Lf, Uf, d, draws=8, seed=sd))
            gains.append(e_id / e_f)
        med[name] = np.median(gains)
        spr[name] = np.median(spreads)
        print(f"{name:16s} spread {spr[name]:9.2e}  gain median {med[name]:.2f}x"
              f"  range [{min(gains):.2f}, {max(gains):.2f}]  seeds {seeds}")
    assert 0.95 < med["geometric k=1e4"] < 1.15
    assert med["wilkinson"] > med["nonneg random"] > med["geometric k=1e4"] - 0.05
    assert 1.3 < med["wilkinson"] < 3.0
    assert spr["wilkinson"] > 100 * spr["geometric k=1e4"]


def test_truncation_share_completes_per_update_accounting():
    # registered prediction (written before running, RTN): the dropped-pair
    # term over the representation error, as a RATIO OF FROBENIUS NORMS, is
    # about 2, matching the record's 10:5 additive split of the per-panel
    # constant 15. band: 2 within a factor 1.5 ([1.33, 3.0]) for geometric
    # and nonneg at depths 3 and 4; wilkinson gets only a wide gate
    # ([0.5, 8]) per its documented volatility.
    fams = {
        "geometric k=1e4": gen_geometric(256, 1e4, seed=5),
        "nonneg random": nonneg(256, seed=1),
        "wilkinson": wilkinson(256),
    }
    seeds = (0, 1, 2)
    ratios = {}
    for name, A in fams.items():
        L21, U12 = panel_step(A, 32)
        for dep in (3, 4):
            reps = [np.sqrt(measured_sq_error(L21, U12, dep, draws=8, seed=sd))
                    for sd in seeds]
            tr = omitted_pairs_error(L21, U12, dep)
            r = tr / np.median(reps)
            ratios[name, dep] = r
            print(f"{name:16s} depth {dep}: rep {np.median(reps):.3e}"
                  f" [{min(reps):.3e}, {max(reps):.3e}] seeds {seeds}"
                  f"  trunc {tr:.3e}  ratio {r:.2f}  (record 10:5 ~ 2)")
    for name in ("geometric k=1e4", "nonneg random"):
        for dep in (3, 4):
            assert 1.33 < ratios[name, dep] < 3.0
    # DOCUMENTED DISAGREEMENT (first run, 2026-09-03): wilkinson's ratio is
    # exactly 0, not in the predicted wide band. its L21 is all +-1, captured
    # entirely by slice 0, so every dropped pair is identically zero and the
    # truncation term vanishes. on lattice-exact panels the split is 0:15,
    # not 10:5; same regime the T4 diagnostic flags. guarded as measured:
    for dep in (3, 4):
        assert ratios["wilkinson", dep] == 0.0