"""Checks for the scale-controlled accumulation experiment (P-B)."""

import numpy as np
from accumulation import error_growth, fit_slope
from sr_slicing import sr_slice_rows, sr_slice_cols

SEEDS = (0, 1, 2)


def slopes_for(slicers=None):
    return [fit_slope(error_growth(k=128, seed=sd, slicers=slicers))
            for sd in SEEDS]


def test_scale_controlled_accumulation_slope():
    # registered prediction (written before running): with scales controlled
    # (iid fresh standard-normal operands each step) the log-log slope of
    # accumulated error versus step count is ~0.5 in BOTH rounding modes,
    # band [0.4, 0.6]. the 0.74 gate-versus-n slope was the composite; if
    # this lands at 0.5 the residual 0.24 belongs to the scale profile.
    s_rtn = slopes_for()
    print(f"RTN: slope median {np.median(s_rtn):.3f}"
          f" [{min(s_rtn):.3f}, {max(s_rtn):.3f}] seeds {SEEDS}"
          "  (composite was 0.74; independence says 0.5)")
    assert 0.4 < np.median(s_rtn) < 0.6

    # SR with the spec's fixed dither key: first run (2026-09-03) measured
    # slope 0.932 [0.932, 0.933] -- a DOCUMENTED DISAGREEMENT with the
    # prediction above. cause: the counter-based dither keyed only on
    # (i, j, level) reuses the identical field every step, and for fixed
    # dither u the conditional error mean is (0.5 - u) quanta, a per-entry
    # bias that accumulates coherently. M1 mean-independence holds within
    # a step, not across steps, under a fixed key. guarded as measured:
    fixed = lambda t: (sr_slice_rows, sr_slice_cols)
    s_fix = slopes_for(fixed)
    print(f"SR fixed key: slope median {np.median(s_fix):.3f}"
          f" [{min(s_fix):.3f}, {max(s_fix):.3f}] seeds {SEEDS}"
          "  (documented disagreement: fixed dither field accumulates bias)")
    assert 0.85 < np.median(s_fix) < 1.05


def test_sr_per_step_key_restores_sqrt_growth():
    # registered prediction (written before running): keying the dither per
    # step (still counter-based, still deterministic) removes the cross-step
    # bias and the slope returns to ~0.5, band [0.4, 0.6]. this isolates the
    # fixed key as the cause of the linear growth above and is the spec fix:
    # the kernel key must include the update/panel index.
    keyed = lambda t: (
        lambda A, d: sr_slice_rows(A, d, key=0x9E3779B9 + 1000003 * (t + 1)),
        lambda B, d: sr_slice_cols(B, d, key=0x51ED2701 + 999983 * (t + 1)),
    )
    s = slopes_for(keyed)
    print(f"SR per-step key: slope median {np.median(s):.3f}"
          f" [{min(s):.3f}, {max(s):.3f}] seeds {SEEDS}")
    assert 0.4 < np.median(s) < 0.6
