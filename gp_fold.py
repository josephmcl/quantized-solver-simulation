"""Certified shared-fold selection for the sliced trailing update.

Operands A (row max-scaled) and B (column max-scaled), fold h on the
contraction index, A' = A diag(h), B' = diag(1/h) B. The expected squared
quantized-product error (Thm 3.3 with uniform fields) is a posynomial in
(h, r, c) with monomial constraints, i.e. a geometric program; in y = log h
it is convex, so the minimizer is certified global. The row/column maxima
are what couple the two operands; the range rule ignores that coupling,
this does not.

The max terms are smoothed with a log-sum-exp of sharpness tau so a plain
quasi-Newton solve works without a GP solver; take tau large and check
the smoothed optimum against the exact objective.
"""

import numpy as np
from scipy.optimize import minimize, linprog

Q = 127.0


def exact_objective(y, A, B):
    h = np.exp(y)
    r = np.max(np.abs(A) * h[None, :], axis=1) / Q
    c = np.max(np.abs(B) / h[:, None], axis=0) / Q
    alpha = np.sum(A**2, axis=0); beta = np.sum(B**2, axis=1)
    R, C, K = np.sum(r**2), np.sum(c**2), A.shape[1]
    return (R * np.sum(beta / h**2) + C * np.sum(alpha * h**2) + K * R * C / 12) / 12


def smooth_objective(y, A, B, tau):
    h = np.exp(y)
    lA = np.log(np.abs(A) + 1e-300) + y[None, :]
    lB = np.log(np.abs(B) + 1e-300) - y[:, None]
    r = np.exp((np.logaddexp.reduce(tau * lA, axis=1)) / tau) / Q
    c = np.exp((np.logaddexp.reduce(tau * lB, axis=0)) / tau) / Q
    alpha = np.sum(A**2, axis=0); beta = np.sum(B**2, axis=1)
    R, C, K = np.sum(r**2), np.sum(c**2), A.shape[1]
    return np.log((R * np.sum(beta / h**2) + C * np.sum(alpha * h**2) + K * R * C / 12) / 12)


def solve_fold(A, B, tau=200.0, y0=None):
    """Returns h*, certified gain F(1)/F(h*) on the exact objective."""
    K = A.shape[1]
    y0 = np.zeros(K) if y0 is None else y0
    res = minimize(smooth_objective, y0, args=(A, B, tau), method="L-BFGS-B")
    y = res.x - np.mean(res.x)          # overall scale of h is a gauge, pin it
    gain = exact_objective(np.zeros(K), A, B) / exact_objective(y, A, B)
    return np.exp(y), gain


def identity_lp_test(A, B, tol=1e-9):
    """Prop 4.3: is the identity fold a stationary point of the exact
    log-domain objective? LP feasibility over tie weights. Returns
    (feasible, max residual). Generic operands have no ties and fail
    unless the gradient happens to vanish; tie-rich operands are the
    interesting case."""
    m, K = A.shape; p = B.shape[1]
    r = np.max(np.abs(A), axis=1) / Q
    c = np.max(np.abs(B), axis=0) / Q
    alpha = np.sum(A**2, axis=0); beta = np.sum(B**2, axis=1)
    R, C, SA, SB = np.sum(r**2), np.sum(c**2), np.sum(alpha), np.sum(beta)
    tieA = np.abs(np.abs(A) / Q - r[:, None]) <= tol * (r[:, None] + 1e-300)
    tieB = np.abs(np.abs(B) / Q - c[None, :]) <= tol * (c[None, :] + 1e-300)
    # variables: lambda_ik (i,k in tieA), mu_jk (j,k in tieB)
    ia = np.argwhere(tieA); ib = np.argwhere(tieB)
    nv = len(ia) + len(ib)
    Aeq, beq = [], []
    for k in range(K):                              # stationarity per k
        row = np.zeros(nv)
        for v, (i, kk) in enumerate(ia):
            if kk == k: row[v] = r[i]**2 * (SB + K * C)
        for v, (j, kk) in enumerate(ib):
            if kk == k: row[len(ia) + v] = -c[j]**2 * (SA + K * R)
        Aeq.append(row); beq.append(beta[k] * R - alpha[k] * C)
    for i in range(m):                              # weights sum to one
        row = np.zeros(nv)
        for v, (ii, kk) in enumerate(ia):
            if ii == i: row[v] = 1
        Aeq.append(row); beq.append(1)
    for j in range(p):
        row = np.zeros(nv)
        for v, (jj, kk) in enumerate(ib):
            if jj == j: row[len(ia) + v] = 1
        Aeq.append(row); beq.append(1)
    res = linprog(np.zeros(nv), A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=[(0, None)] * nv, method="highs")
    resid = np.max(np.abs(np.array(Aeq) @ res.x - beq)) if res.success else np.inf
    return res.success and resid < 1e-6, resid


if __name__ == "__main__":
    # toy check only: the anti-correlated pair the range rule handled,
    # the GP should do at least as well. our panels are the real question.
    from quantize import range_rule, apply_contraction
    rng = np.random.default_rng(0)
    A = rng.standard_normal((96, 96)); B = rng.standard_normal((96, 96))
    s = np.logspace(-3, 3, 96); A2, B2 = A * s[None, :], B / s[:, None]
    h_gp, gain_gp = solve_fold(A2, B2)
    y_rr = np.log(range_rule(A2, B2)); y_rr -= y_rr.mean()
    gain_rr = exact_objective(np.zeros(96), A2, B2) / exact_objective(y_rr, A2, B2)
    print(f"predicted-error gain, range rule {gain_rr:.1f}x   GP {gain_gp:.1f}x")
    feas, resid = identity_lp_test(A2, B2)
    print(f"identity stationary on this toy: {feas}")