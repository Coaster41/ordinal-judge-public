"""Verification for `recalibrate.py`'s pmf/cdf recalibration families.

Run: python tests/test_recalibrate.py      (or: pytest tests/)
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from ojc import losses as Lo
from ojc import metrics as mt
from ojc import recalibrate as RC

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def rand_pmf(rng, n, K):
    p = rng.gamma(1.0, size=(n, K))
    return p / p.sum(1, keepdims=True)


def rand_counts(rng, pmf, M):
    return np.stack([rng.multinomial(M, p) for p in pmf]).astype(np.float64)


# --------------------------------------------------------------------------- #
print("\n1. T=1, b=0 is the exact identity in both families")
# cdf: exact to float64 precision (its clip epsilon never engages away from
# 0/1). pmf: exact up to the fixed additive smoothing alpha it applies before
# every log, by construction (see module docstring) - so its tolerance is
# O(K * alpha), not machine precision.
rng = np.random.RandomState(0)
p = rand_pmf(rng, 200, 6)
tol = {"pmf": 20 * RC.ALPHA_SMOOTH, "cdf": 1e-9}
for family in ("pmf", "cdf"):
    for n_params in (1, 2):
        theta = np.zeros(n_params)  # T = exp(0) = 1, b = 0
        out = RC.apply(p, theta, family, n_params).numpy()
        err = np.abs(out - p).max()
        check(f"identity: family={family} n_params={n_params}", err < tol[family],
              f"max|p'-p|={err:.2e}  (tol={tol[family]:.1e})")

# --------------------------------------------------------------------------- #
print("\n2. 2-param variants strictly contain the 1-param ones")
# Same fold, same objective: the 2-param fit's objective must be <= the
# 1-param fit's, since gamma=0 (pmf) / b=0 (cdf) is a feasible point of the
# wider family.
rng = np.random.RandomState(1)
p = rand_pmf(rng, 300, 5)
truth = rand_pmf(rng, 1, 5)[0]
counts = rand_counts(rng, np.tile(truth, (300, 1)), 12)
for family in ("pmf", "cdf"):
    for objective in ("ce", "rps"):
        r1 = RC.fit(p, counts, family, 1, objective)
        r2 = RC.fit(p, counts, family, 2, objective)
        check(f"nesting: family={family} objective={objective}",
              r2["loss"] <= r1["loss"] + 1e-6,
              f"1-param={r1['loss']:.6f} 2-param={r2['loss']:.6f}")

# --------------------------------------------------------------------------- #
print("\n3. CE-fitting on a converged, unregularised, self-consistent prediction "
     "recovers T-hat ~= 1 (nested)")
# If pred == the empirical distribution used as target (what an unregularised
# CE-optimal, full-capacity softmax fit converges to), no further scaling can
# reduce CE further, so the CE-optimal recalibration is the identity.
rng = np.random.RandomState(2)
n, K = 500, 5
pmf_true = rand_pmf(rng, n, K)
counts = rand_counts(rng, pmf_true, 4000)  # huge M -> phat ~= pmf_true, ~no MC noise
for family in ("pmf", "cdf"):
    for n_params in (1, 2):
        res = RC.fit(pmf_true, counts, family, n_params, "ce")
        T_hat = res["params"]["T"]
        b_hat = res["params"]["b"]
        dev = abs(T_hat - 1.0)
        ok = dev < 0.05
        check(f"T-hat~=1: family={family} n_params={n_params}", ok,
              f"T_hat={T_hat:.4f} b_hat={b_hat:.4f}"
              + ("" if ok else "  [WARNING: >5% deviation from nested T=1]"))

# --------------------------------------------------------------------------- #
print("\n4. Double-softmax flag")
K = 5
thr = RC.double_softmax_threshold(K)
check("threshold formula", abs(thr - np.e / (K - 1 + np.e)) < 1e-12)
check("flags a scale below threshold", RC.flag_double_softmax(thr * 0.5, K) is True)
check("does not flag a scale at/above threshold",
      RC.flag_double_softmax(thr * 1.5, K) is False)
check("does not flag identity (T=1)", RC.flag_double_softmax(1.0, K) is False)

# --------------------------------------------------------------------------- #
print("\n5. CLM equivalence: cdf-domain T rescales the learned log-sigma exactly")


def clm_pmf(alpha, f, s):
    z = (alpha[None, :] - f[:, None]) / s
    cum = 1.0 / (1.0 + np.exp(-z))
    cum_full = np.concatenate([np.zeros((len(f), 1)), cum, np.ones((len(f), 1))], axis=1)
    return np.diff(cum_full, axis=1)


rng = np.random.RandomState(3)
n, K = 400, 6
alpha = np.sort(rng.normal(0, 1.5, size=K - 1))
f = rng.normal(0, 1.0, size=n)
s0 = 1.0
p_s0 = clm_pmf(alpha, f, s0)

# (a) pure algebraic identity: applying cdf-family theta=(log(T),0) to the
# s0-scale pmf must exactly reproduce the CLM pmf at scale s0*T.
for T_true in (0.6, 1.0, 1.7):
    p_sT_direct = clm_pmf(alpha, f, s0 * T_true)
    p_sT_recal = RC.apply(p_s0, [np.log(T_true), 0.0], "cdf", 2).numpy()
    err = np.abs(p_sT_direct - p_sT_recal).max()
    check(f"algebraic: cdf-recal(T={T_true}) == CLM at rescaled log-sigma", err < 1e-8,
          f"max abs diff={err:.2e}")

# (b) fitted recovery: counts drawn (at very high M, so phat ~= population)
# from the CLM at a DIFFERENT true scale; CE-fitting the 1-param cdf family
# on the s0 predictions should recover T-hat ~= s_true/s0.
T_true = 1.8
p_true = clm_pmf(alpha, f, s0 * T_true)
counts = np.round(p_true * 20000)
res = RC.fit(p_s0, counts, "cdf", 1, "ce")
T_hat = res["params"]["T"]
dev = abs(T_hat - T_true) / T_true
check("fitted T-hat matches the implied CLM scale change", dev < 0.02,
      f"T_hat={T_hat:.4f} T_true={T_true:.4f} rel_dev={dev:.4f}")

# --------------------------------------------------------------------------- #
print("\n6. Cross-fit pools every item exactly once, from a fit that never saw it")
rng = np.random.RandomState(4)
n, K = 253, 5  # not a multiple of n_folds, to exercise uneven fold sizes
p = rand_pmf(rng, n, K)
counts = rand_counts(rng, rand_pmf(rng, n, K), 10)
cf = RC.cross_fit(p, counts, "pmf", 2, "ce", n_folds=5, seed=0)
check("pooled predictions cover every item", cf["pooled"].shape == (n, K))
check("pooled rows are valid pmfs",
      np.allclose(cf["pooled"].sum(-1), 1.0, atol=1e-6))
check("fold sizes are balanced to within 1",
      np.ptp(np.bincount(cf["fold_id"])) <= 1,
      f"{np.bincount(cf['fold_id'])}")
check("one theta per fold", cf["thetas"].shape == (5, 2))

# recalibrating with the fold-specific theta on its own held-out rows must
# match what the pooled array actually stored there (no accidental leakage
# of a different fold's fit into these rows)
mismatch = 0.0
for f in range(5):
    held = cf["fold_id"] == f
    direct = RC.apply(p[held], cf["thetas"][f], "pmf", 2).numpy()
    mismatch = max(mismatch, np.abs(direct - cf["pooled"][held]).max())
check("pooled rows equal apply(theta_fold) on their own held-out fold",
      mismatch < 1e-10, f"max diff={mismatch:.2e}")

# --------------------------------------------------------------------------- #
print("\n7. Cross-fitting improves (or does not worsen) held-out CE relative to "
     "the unrecalibrated input, on a miscalibrated synthetic input")
rng = np.random.RandomState(5)
n, K = 600, 5
truth = rand_pmf(rng, n, K)
counts = rand_counts(rng, truth, 8)
# manufacture an overconfident (sharpened) predictor from the truth
over = truth ** 3
over = over / over.sum(-1, keepdims=True)
cf = RC.cross_fit(over, counts, "cdf", 1, "ce", n_folds=5, seed=0)
ce_pre = mt.soft_ce(over, counts).mean()
ce_post = mt.soft_ce(cf["pooled"], counts).mean()
check("cross-fit recalibration reduces held-out CE on an overconfident input",
      ce_post <= ce_pre + 1e-6, f"pre={ce_pre:.4f} post={ce_post:.4f}")

# --------------------------------------------------------------------------- #
print("\n8. apply() validates its inputs")


def _raises(fn):
    try:
        fn()
    except (ValueError,):
        return True
    return False


check("unknown family raises", _raises(lambda: RC.apply(p[:5], [0.0], "bogus", 1)))
check("wrong theta length raises",
      _raises(lambda: RC.apply(p[:5], [0.0, 0.0, 0.0], "pmf", 2)))
check("n_folds < 2 raises",
      _raises(lambda: RC.cross_fit(p, counts, "pmf", 1, "ce", n_folds=1)))

# --------------------------------------------------------------------------- #
print("\n9. describe() names beta and temperature unambiguously")
d_pmf = RC.describe(np.array([np.log(2.0)]), "pmf", 1)
check("pmf: beta = exp(theta0)", abs(d_pmf["beta"] - 2.0) < 1e-12)
check("pmf: temperature = 1/beta", abs(d_pmf["temperature"] - 0.5) < 1e-12)
d_cdf = RC.describe(np.array([np.log(2.0), 0.3]), "cdf", 2)
check("cdf: temperature = exp(theta0), shift = theta1",
      abs(d_cdf["temperature"] - 2.0) < 1e-12 and abs(d_cdf["shift_b"] - 0.3) < 1e-12)
# an overconfident predictor must come back with temperature > 1 (flatten)
rng9 = np.random.RandomState(9)
p9 = rand_pmf(rng9, 2000, 5)
c9 = rand_counts(rng9, p9, 1)
sharp = p9 ** 3 / (p9 ** 3).sum(1, keepdims=True)
fit9 = RC.fit(sharp, c9, "pmf", 1, "ce")
T9 = RC.describe(fit9["theta"], "pmf", 1)["temperature"]
check("overconfident at M=1 -> fitted temperature near 3", 2.4 < T9 < 3.6, f"T={T9:.3f}")


# --------------------------------------------------------------------------- #
def test_recalibrate():
    assert not FAIL, FAIL


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        sys.exit(1)
    print("pmf/cdf recalibration families verified")
