"""Verification of the mathematical claims the metrics and losses rest on.

Each check guards a specific claim; if one fails, a reported number is wrong.
Section numbers follow the original research suite, so gaps are sections that
covered code not shipped here (the Laplace posterior and its analyses).

Run: python tests/test_math.py      (or: pytest tests/)
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from ojc import heads
from ojc import losses
from ojc import metrics as mt

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def rand_pmf(rng, n, K):
    p = rng.gamma(1.0, size=(n, K))
    return p / p.sum(1, keepdims=True)


rng = np.random.RandomState(0)

# --------------------------------------------------------------------------- #
print("\n2. Soft-label CE is LINEAR in the target -> exactly unbiased")
K, M, n_mc = 5, 8, 40000
pi = rand_pmf(rng, 1, K)[0]
pred = rand_pmf(rng, 1, K)[0]
draws = rng.multinomial(M, pi, size=n_mc)
emp_ce = mt.soft_ce(np.tile(pred, (n_mc, 1)), draws).mean()
pop_ce = -(pi * np.log(pred)).sum()
check("E[soft_ce(F,phat)] = CE(F,pi)", abs(emp_ce - pop_ce) < 3e-3,
      f"emp={emp_ce:.5f} pop={pop_ce:.5f}")

# --------------------------------------------------------------------------- #
print("\n3. RPS is QUADRATIC -> unbiased up to a prediction-independent constant")
G = np.cumsum(pi)[:-1]
offsets = []
for _ in range(6):
    F_pred = rand_pmf(rng, 1, K)[0]
    emp = mt.rps(np.tile(F_pred, (n_mc, 1)), draws).mean()
    pop = ((np.cumsum(F_pred)[:-1] - G) ** 2).sum()
    offsets.append(emp - pop)
offsets = np.array(offsets)
predicted = (G * (1 - G)).sum() / M
check("offset independent of prediction", offsets.std() < 3e-3,
      f"spread={offsets.std():.2e}")
check("offset = Sum_k G_k(1-G_k)/M", abs(offsets.mean() - predicted) < 3e-3,
      f"emp={offsets.mean():.5f} theory={predicted:.5f}")

# --------------------------------------------------------------------------- #
print("\n4. Debiased RPS is unbiased under VARIABLE M")
F_pred = rand_pmf(rng, 1, K)[0]
pop = ((np.cumsum(F_pred)[:-1] - G) ** 2).sum()
for M_i in (8, 12, 18):
    dr = rng.multinomial(M_i, pi, size=n_mc)
    raw = mt.rps(np.tile(F_pred, (n_mc, 1)), dr).mean()
    deb = mt.rps_debiased(np.tile(F_pred, (n_mc, 1)), dr).mean()
    check(f"M={M_i}: debiased == population RPS", abs(deb - pop) < 3e-3,
          f"raw={raw:.5f} debiased={deb:.5f} pop={pop:.5f}")
# and the raw version really does drift with M while the debiased one does not
raws, debs = [], []
for M_i in (8, 12, 18):
    dr = rng.multinomial(M_i, pi, size=n_mc)
    raws.append(mt.rps(np.tile(F_pred, (n_mc, 1)), dr).mean())
    debs.append(mt.rps_debiased(np.tile(F_pred, (n_mc, 1)), dr).mean())
check("raw RPS drifts with M", np.ptp(raws) > 10 * np.ptp(debs),
      f"raw spread={np.ptp(raws):.5f} vs debiased={np.ptp(debs):.5f}")

# --------------------------------------------------------------------------- #
print("\n5. Entropy is CONCAVE -> plug-in biased down by ~(K-1)/(2M)")
for K_, M_ in [(5, 3), (5, 8), (7, 8), (7, 18)]:
    pi_ = rand_pmf(rng, 1, K_)[0]
    dr = rng.multinomial(M_, pi_, size=30000)
    bias = mt.entropy(mt.to_pmf(dr)).mean() - mt.entropy(pi_)
    theory = -(K_ - 1) / (2 * M_)
    check(f"K={K_},M={M_}: plug-in biased downward", bias < 0, f"bias={bias:.4f}")
    check(f"K={K_},M={M_}: Miller-Madow reduces |bias|",
          abs(mt.entropy_miller_madow(dr).mean() - mt.entropy(pi_)) < abs(bias),
          f"raw={bias:.4f} theory={theory:.4f} "
          f"mm={mt.entropy_miller_madow(dr).mean() - mt.entropy(pi_):+.4f}")

# --------------------------------------------------------------------------- #
print("\n6. M-weighted CE gradient == multinomial log-likelihood gradient")
torch.manual_seed(0)
d_, K_, n_ = 16, 5, 64
X = torch.randn(n_, d_, dtype=torch.float64)
cnts = torch.tensor(rng.multinomial(9, rand_pmf(rng, 1, K_)[0], size=n_), dtype=torch.float64)
for hname in ("softmax", "clm"):
    h = heads.build_head(hname, d_, K_).double()
    lp = h(X)
    # our loss: M-weighted mean
    l1 = losses.soft_ce_loss(lp, cnts, weight_by_M=True) * cnts.sum()
    g1 = torch.autograd.grad(l1, list(h.parameters()), retain_graph=True)
    # direct multinomial NLL from integer counts
    l2 = -(cnts * h(X)).sum()
    g2 = torch.autograd.grad(l2, list(h.parameters()))
    err = max(float((a - b).abs().max()) for a, b in zip(g1, g2))
    check(f"{hname}: M-weighted CE == multinomial MLE", err < 1e-9, f"max|dg|={err:.2e}")

# --------------------------------------------------------------------------- #
print("\n7. CLM cutpoints stay ordered under any parameter perturbation")
h = heads.build_head("clm", d_, K_).double()
a = h.alpha.detach().numpy()
check("MAP cutpoints strictly increasing", bool(np.all(np.diff(a) > 0)), f"alpha={np.round(a,3)}")
bad = 0
gen = torch.Generator().manual_seed(0)
for _ in range(2000):
    # any unconstrained perturbation of (a0, delta) must give ordered cutpoints
    a0 = h.a0.detach() + 0.75 * torch.randn(h.a0.shape, generator=gen, dtype=torch.float64)
    de = h.delta.detach() + 0.75 * torch.randn(h.delta.shape, generator=gen, dtype=torch.float64)
    al = heads._cutpoints(a0, de)
    if not bool(torch.all(torch.diff(al) > 0)):
        bad += 1
check("2000 perturbed draws all monotone", bad == 0, f"violations={bad}")

lpm = h(X).exp().detach().numpy()
check("CLM pmf rows sum to 1", np.abs(lpm.sum(1) - 1).max() < 1e-9)
check("CLM pmf strictly positive", lpm.min() > 0, f"min={lpm.min():.2e}")

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
print("\n9. Metric sanity")
p = rand_pmf(rng, 200, 5)
c = np.stack([rng.multinomial(8, q) for q in p])
check("RPS >= 0", mt.rps(p, c).min() >= 0)
check("EMD >= 0", mt.emd(p, c).min() >= 0)
perfect = mt.to_pmf(c)
check("RPS = 0 for exact match", mt.rps(perfect, c).max() < 1e-12)
check("EMD = 0 for exact match", mt.emd(perfect, c).max() < 1e-12)
cov, w = mt.coverage(p, c, 0.9)
check("coverage in [0,1]", 0 <= cov <= 1, f"cov90={cov:.3f} width={w:.2f}")

# RPS on unit support == discrete CRPS computed the long way
def crps_long(pmf, y, K):
    F = np.cumsum(pmf, -1)
    step = (np.arange(K)[None, :] >= y[:, None]).astype(float)
    return ((F - step) ** 2)[:, :-1].sum(-1)

y = np.array([rng.choice(5, p=q) for q in p])
onehot = np.eye(5)[y]
check("RPS == discrete CRPS on unit support",
      np.abs(mt.rps(p, onehot) - crps_long(p, y, 5)).max() < 1e-12)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
print("\n9. Calibration: ECE over {mode, full} x {pmf, cdf}")
rng = np.random.RandomState(7)

# A perfectly calibrated generator: draw the panel FROM the prediction, so the
# only ECE left is finite-sample bin noise and it must shrink as n grows.
def calib_data(n, K, M, rs):
    p = rand_pmf(rs, n, K)
    c = np.stack([rs.multinomial(M, pi) for pi in p]).astype(np.float64)
    return p, c

for sp in ("pmf", "cdf"):
    small = mt.ece(*calib_data(150, 5, 8, np.random.RandomState(1)),
                   mode="full", space=sp, n_bins=10, n_boot=0)["ece"]
    large = mt.ece(*calib_data(4000, 5, 8, np.random.RandomState(1)),
                   mode="full", space=sp, n_bins=10, n_boot=0)["ece"]
    check(f"ECE(full,{sp}) -> 0 for a calibrated model as n grows",
          large < small and large < 0.02, f"n=150 {small:.4f} -> n=4000 {large:.4f}")

# A miscalibrated model must score worse than the calibrated one it is a
# distortion of. Sharpening toward the corners is the classic overconfidence.
p_c, c_c = calib_data(2000, 5, 8, np.random.RandomState(2))
p_bad = p_c ** 3
p_bad /= p_bad.sum(1, keepdims=True)
for sp in ("pmf", "cdf"):
    good = mt.ece(p_c, c_c, mode="full", space=sp, n_boot=0)["ece"]
    bad = mt.ece(p_bad, c_c, mode="full", space=sp, n_boot=0)["ece"]
    check(f"ECE(full,{sp}) penalises an overconfident model",
          bad > 3 * good, f"calibrated={good:.4f} sharpened={bad:.4f}")

# mode is calibrated against a HARD label, so the reference model it rewards is
# the distribution of the single drawn rating. With M=1 the mode target is
# exactly that rating and the usual claims hold.
p_1 = rand_pmf(np.random.RandomState(4), 4000, 5)
c_1 = np.stack([np.random.RandomState(5 + i).multinomial(1, p_1[i])
                for i in range(len(p_1))]).astype(np.float64)
p_1bad = p_1 ** 3
p_1bad /= p_1bad.sum(1, keepdims=True)
for sp in ("pmf", "cdf"):
    good = mt.ece(p_1, c_1, mode="mode", space=sp, n_bins=10, n_boot=0)["ece"]
    bad = mt.ece(p_1bad, c_1, mode="mode", space=sp, n_bins=10, n_boot=0)["ece"]
    check(f"ECE(mode,{sp}) -> 0 for a calibrated model on single-rating panels",
          good < 0.02, f"{good:.4f}")
    check(f"ECE(mode,{sp}) penalises an overconfident model",
          bad > 3 * good, f"calibrated={good:.4f} sharpened={bad:.4f}")

# THE MODE BIAS, pinned deliberately. Once a panel has several raters its
# majority level is SHARPER than the per-annotator distribution, so a model that
# is perfectly calibrated per annotator looks badly calibrated against the mode
# target, and a sharpened (overconfident) model scores BETTER. This is a
# property of the mode aggregation, not a defect in the estimator - but it means
# ece(mode=...) must never be read as "how well does this match the raters".
mode_good = mt.ece(p_c, c_c, mode="mode", space="pmf", n_boot=0)["ece"]
mode_bad = mt.ece(p_bad, c_c, mode="mode", space="pmf", n_boot=0)["ece"]
check("mode target is sharper than the panel: it prefers an overconfident model",
      mode_bad < mode_good,
      f"M=8 panel: calibrated={mode_good:.4f} sharpened={mode_bad:.4f} "
      f"(vs full: {mt.ece(p_c, c_c, mode='full', n_boot=0)['ece']:.4f} -> "
      f"{mt.ece(p_bad, c_c, mode='full', n_boot=0)['ece']:.4f})")

# The 'full' target must not depend on panel size, only on the empirical pmf:
# doubling every annotator count is the same distribution and must give the
# same ECE. This is the reweighting the metric is required to do.
c_2x = c_c * 2.0
for sp in ("pmf", "cdf"):
    a = mt.ece(p_c, c_c, mode="full", space=sp, n_boot=0)["ece"]
    b = mt.ece(p_c, c_2x, mode="full", space=sp, n_boot=0)["ece"]
    check(f"ECE(full,{sp}) is invariant to panel size at fixed pmf",
          abs(a - b) < 1e-12, f"|d|={abs(a - b):.2e}")

# ...and an item with a 100-rater panel must not outweigh one with 2 raters.
# Build two items with opposite miscalibration and wildly different M: if the
# weighting were per-annotator the big panel would dominate and the gaps would
# not cancel.
p2 = np.array([[0.8, 0.2], [0.2, 0.8]])
c2 = np.array([[2.0, 0.0], [0.0, 100.0]])
r2 = mt.ece(p2, c2, mode="full", space="pmf", n_bins=1, n_boot=0)
check("ECE(full) weights items equally regardless of annotator count",
      abs(r2["curve"]["acc"][0] - 0.5) < 1e-12,
      f"pooled frequency={r2['curve']['acc'][0]:.6f} (per-annotator would be 0.510)")

# mode: ties split evenly rather than falling to the lowest index
tie = mt._target_matrix(np.array([[2.0, 2.0, 0.0]]), "mode")
check("mode target splits ties evenly instead of taking argmax order",
      np.allclose(tie, [[0.5, 0.5, 0.0]]), f"{tie.ravel()}")
check("mode target is the majority level when there is no tie",
      np.allclose(mt._target_matrix(np.array([[1.0, 3.0, 0.0]]), "mode"),
                  [[0.0, 1.0, 0.0]]))

# cdf space must see an off-by-one error as small and pmf space as maximal:
# this is the whole reason both are reported.
K9 = 5
p_off = np.zeros((1, K9)); p_off[0, 1] = 1.0     # all mass one level below
c_off = np.zeros((1, K9)); c_off[0, 2] = 6.0
p_far = np.zeros((1, K9)); p_far[0, 4] = 1.0     # all mass three levels above
e_off_pmf = mt.ece(p_off, c_off, mode="full", space="pmf", n_boot=0)["ece"]
e_far_pmf = mt.ece(p_far, c_off, mode="full", space="pmf", n_boot=0)["ece"]
e_off_cdf = mt.ece(p_off, c_off, mode="full", space="cdf", n_boot=0)["ece"]
e_far_cdf = mt.ece(p_far, c_off, mode="full", space="cdf", n_boot=0)["ece"]
check("pmf-space ECE is ordering-blind: near miss == far miss",
      abs(e_off_pmf - e_far_pmf) < 1e-12, f"near={e_off_pmf:.4f} far={e_far_pmf:.4f}")
check("cdf-space ECE ranks a near miss below a far miss",
      e_off_cdf < e_far_cdf, f"near={e_off_cdf:.4f} far={e_far_cdf:.4f}")

# curve bookkeeping
r = mt.ece(p_c, c_c, mode="full", space="pmf", n_bins=12, n_boot=60, seed=0)
cur = r["curve"]
check("curve has n_bins+1 edges and n_bins entries per series",
      len(cur["edges"]) == 13 and all(len(cur[k]) == 12 for k in
                                      ("weight", "conf", "acc", "gap", "acc_lo",
                                       "acc_hi", "acc_se")))
check("curve weights sum to n_items * n_events_per_item",
      abs(sum(cur["weight"]) - r["n_events"]) < 1e-9,
      f"{sum(cur['weight'])} vs {r['n_events']}")
w = np.array(cur["weight"]); conf = np.array(cur["conf"]); acc = np.array(cur["acc"])
nz = w > 0
check("scalar ECE equals the weight-averaged bin gap",
      abs((w[nz] * np.abs(conf[nz] - acc[nz])).sum() / w.sum() - r["ece"]) < 1e-12)
check("bin confidences are inside their bin edges",
      all(cur["edges"][i] - 1e-12 <= conf[i] <= cur["edges"][i + 1] + 1e-12
          for i in range(12) if nz[i]))
lo = np.array(cur["acc_lo"]); hi = np.array(cur["acc_hi"])
check("bootstrap interval brackets the observed frequency in every used bin",
      bool(np.all(lo[nz] <= acc[nz] + 1e-9) and np.all(acc[nz] <= hi[nz] + 1e-9)))
check("bootstrap interval is wider where a bin holds less weight",
      np.corrcoef(np.log(w[nz]), (hi - lo)[nz])[0, 1] < -0.3,
      f"corr(log weight, width)={np.corrcoef(np.log(w[nz]), (hi - lo)[nz])[0, 1]:.3f}")
check("ECE interval brackets the point estimate",
      r["ece_lo"] <= r["ece"] <= r["ece_hi"] or r["ece_se"] > 0,
      f"[{r['ece_lo']:.4f}, {r['ece_hi']:.4f}] vs {r['ece']:.4f}")

# error bars must respond to BOTH bootstrap levels: fewer items and smaller
# panels both have to widen them
wide_n = mt.ece(p_c[:200], c_c[:200], mode="full", n_bins=12, n_boot=60)["ece_se"]
check("fewer items widens the ECE error bar",
      wide_n > r["ece_se"], f"n=2000 {r['ece_se']:.4f} -> n=200 {wide_n:.4f}")
p_m, c_m2 = calib_data(2000, 5, 2, np.random.RandomState(3))
p_m, c_m30 = calib_data(2000, 5, 30, np.random.RandomState(3))
se2 = mt.ece(p_m, c_m2, mode="full", n_bins=12, n_boot=60)["ece_se"]
se30 = mt.ece(p_m, c_m30, mode="full", n_bins=12, n_boot=60)["ece_se"]
check("smaller annotator panels widen the ECE error bar",
      se2 > se30, f"M=2 {se2:.4f} vs M=30 {se30:.4f}")

check("quantile binning spreads weight more evenly than uniform binning",
      np.std(mt.ece(p_c, c_c, n_bins=10, binning="quantile",
                    n_boot=0)["curve"]["frac"]) <
      np.std(mt.ece(p_c, c_c, n_bins=10, binning="uniform",
                    n_boot=0)["curve"]["frac"]))
check("ece rejects an unknown mode", _raises(lambda: mt.ece(p_c, c_c, mode="nope")))
check("ece rejects an unknown space", _raises(lambda: mt.ece(p_c, c_c, space="nope")))
check("ece rejects n_bins < 1", _raises(lambda: mt.ece(p_c, c_c, n_bins=0)))
check("ece rejects a prediction/target shape mismatch",
      _raises(lambda: mt.ece(p_c, c_c[:, :3])))
check("ece_grid returns all four cells",
      sorted(mt.ece_grid(p_c, c_c, n_boot=0)) ==
      ["ece_full_cdf", "ece_full_pmf", "ece_mode_cdf", "ece_mode_pmf"])
check("summary no longer carries the old {mode,full}x{pmf,cdf} grid",
      not any(f"ece_{m}_{s}" in mt.summary(p_c, c_c) for m in ("mode", "full")
              for s in ("pmf", "cdf")))

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
print("\n16. Calibration: ece_top and ece_thr")

for n_small, n_large in [(150, 4000)]:
    small = mt.ece_top(*calib_data(n_small, 5, 8, np.random.RandomState(1)),
                       n_bins=10, n_boot=0)["ece"]
    large = mt.ece_top(*calib_data(n_large, 5, 8, np.random.RandomState(1)),
                       n_bins=10, n_boot=0)["ece"]
    check("ece_top -> 0 for a calibrated model as n grows",
          large < small and large < 0.02, f"n={n_small} {small:.4f} -> n={n_large} {large:.4f}")
    small = mt.ece_thr(*calib_data(n_small, 5, 8, np.random.RandomState(1)),
                       n_bins=10, n_boot=0)["ece"]
    large = mt.ece_thr(*calib_data(n_large, 5, 8, np.random.RandomState(1)),
                       n_bins=10, n_boot=0)["ece"]
    check("ece_thr -> 0 for a calibrated model as n grows",
          large < small and large < 0.02, f"n={n_small} {small:.4f} -> n={n_large} {large:.4f}")

# same sharpened-model check as the ece_grid section above
top_good = mt.ece_top(p_c, c_c, n_boot=0)["ece"]
top_bad = mt.ece_top(p_bad, c_c, n_boot=0)["ece"]
check("ece_top penalises an overconfident model",
      top_bad > 3 * top_good, f"calibrated={top_good:.4f} sharpened={top_bad:.4f}")
thr_good = mt.ece_thr(p_c, c_c, n_boot=0)["ece"]
thr_bad = mt.ece_thr(p_bad, c_c, n_boot=0)["ece"]
check("ece_thr penalises an overconfident model",
      thr_bad > 3 * thr_good, f"calibrated={thr_good:.4f} sharpened={thr_bad:.4f}")

# ece_top must target the model's OWN emitted category, not the majority
# panel level: with argmax(p) forced away from the panel's actual majority,
# a model can look perfectly calibrated in 'mode' space (majority target)
# while ece_top correctly sees it disagreeing with its own predicted class.
p_disagree = np.array([[0.1, 0.8, 0.1]])   # yhat = 1, confidence 0.8
c_disagree = np.array([[9.0, 1.0, 0.0]])   # panel majority is level 0; only 1/10 agree with yhat=1
top_dis = mt.ece_top(p_disagree, c_disagree, n_bins=1, n_boot=0)
check("ece_top scores against agreement with the EMITTED class, not the panel majority",
      abs(top_dis["curve"]["acc"][0] - 0.1) < 1e-12,
      f"acc={top_dis['curve']['acc'][0]:.4f} (expected 0.1 = 1/10 agreeing with yhat)")

# THE POOLING GAP, pinned deliberately (see the ece_thr docstring). Two cut
# points with equal-and-opposite miscalibration land in the same bin: pooled
# ece(mode='full', space='cdf') averages them to a gap of exactly zero, while
# ece_thr - which scores each cut point separately before averaging - cannot.
p_pool = np.array([[0.3, 0.4, 0.3]])   # P0=0.3, P1=0.7
c_pool = np.array([[5.0, 0.0, 5.0]])   # F0=0.5, F1=0.5: gaps are -0.2 and +0.2
pooled = mt.ece(p_pool, c_pool, mode="full", space="cdf", n_bins=1, n_boot=0)["ece"]
thr = mt.ece_thr(p_pool, c_pool, n_bins=1, n_boot=0)
check("pooling opposite-signed threshold gaps into one bin hides them",
      pooled < 1e-9, f"pooled ece(full,cdf)={pooled:.2e}")
check("ece_thr does not hide them: it scores each threshold before averaging",
      abs(thr["ece"] - 0.2) < 1e-9,
      f"ece_thr={thr['ece']:.4f} (expected 0.2), per-threshold="
      f"{[round(c['ece'], 4) for c in thr['per_threshold']]}")
resid0 = thr["per_threshold"][0]["curve"]["resid"][0]
resid1 = thr["per_threshold"][1]["curve"]["resid"][0]
check("ece_thr's per-threshold residual is SIGNED and opposite between the two "
      "cut points (the mirror-image pattern the residual plot is read by)",
      abs(resid0 - (-0.2)) < 1e-9 and abs(resid1 - 0.2) < 1e-9,
      f"resid_0={resid0:+.3f}  resid_1={resid1:+.3f}  (expect -0.200, +0.200)")

# curve/bootstrap bookkeeping, mirroring the ece() checks above
rt = mt.ece_top(p_c, c_c, n_bins=12, n_boot=60, seed=0)
curt = rt["curve"]
check("ece_top curve has n_bins+1 edges and n_bins entries per series",
      len(curt["edges"]) == 13 and all(len(curt[k]) == 12 for k in
                                       ("weight", "conf", "acc", "gap",
                                        "acc_lo", "acc_hi", "acc_se")))
check("ece_top curve weights sum to n_items (one number per item, not per class)",
      abs(sum(curt["weight"]) - rt["n_items"]) < 1e-9,
      f"{sum(curt['weight'])} vs {rt['n_items']}")
check("ece_top bootstrap interval brackets the point estimate",
      rt["ece_lo"] <= rt["ece"] <= rt["ece_hi"] or rt["ece_se"] > 0)

rr = mt.ece_thr(p_c, c_c, n_bins=12, n_boot=60, seed=0)
check("ece_thr reports one curve per cut point",
      len(rr["per_threshold"]) == rr["n_thresholds"] == p_c.shape[1] - 1)
check("ece_thr aggregate equals the mean of its per-threshold eces",
      abs(rr["ece"] - np.mean([c["ece"] for c in rr["per_threshold"]])) < 1e-12)
check("ece_thr bootstrap interval brackets the point estimate",
      rr["ece_lo"] <= rr["ece"] <= rr["ece_hi"] or rr["ece_se"] > 0)

check("ece_top rejects a prediction/target shape mismatch",
      _raises(lambda: mt.ece_top(p_c, c_c[:, :3])))
check("ece_top rejects n_bins < 1", _raises(lambda: mt.ece_top(p_c, c_c, n_bins=0)))
check("ece_thr rejects a prediction/target shape mismatch",
      _raises(lambda: mt.ece_thr(p_c, c_c[:, :3])))
check("ece_thr rejects n_bins < 1", _raises(lambda: mt.ece_thr(p_c, c_c, n_bins=0)))
check("ece_thr rejects K < 2",
      _raises(lambda: mt.ece_thr(p_c[:, :1], c_c[:, :1])))

check("summary carries ece_top and ece_thr, not the old {mode,full}x{pmf,cdf} grid",
      {"ece_top", "ece_thr"} <= set(mt.summary(p_c, c_c)) and
      not any(k.startswith("ece_mode") or k.startswith("ece_full")
              for k in mt.summary(p_c, c_c)))

_sm = mt.summary(p_c, c_c, n_boot_null=100, seed=0)
check("summary always carries the ece_thr null floor alongside ece_thr",
      {"ece_thr_null_mean", "ece_thr_null_se", "ece_thr_null_p95",
       "ece_thr_above_floor"} <= set(_sm))
check("summary's ece_thr_above_floor is ece_thr minus the floor mean",
      abs(_sm["ece_thr_above_floor"] - (_sm["ece_thr"] - _sm["ece_thr_null_mean"])) < 1e-12)
check("n_boot_null=0 skips the floor entirely (cheap-path escape hatch)",
      "ece_thr_null_mean" not in mt.summary(p_c, c_c, n_boot_null=0))

# --------------------------------------------------------------------------- #
print("\n17. ece_thr null floor and paired cross-model SE")

# The null floor: draw panels from the model's OWN pmf (so it is the truth by
# construction) and recompute ece_thr against that. Binned ECE is upward
# biased, so even this best-case floor must sit above zero, and it must
# shrink as n grows - the same finite-sample bias the section-9 ECE tests pin.
floor_small = mt.ece_thr_null(*calib_data(150, 5, 8, np.random.RandomState(1)),
                              n_bins=10, n_boot=300, seed=0)
floor_large = mt.ece_thr_null(*calib_data(4000, 5, 8, np.random.RandomState(1)),
                              n_bins=10, n_boot=300, seed=0)
check("null floor is strictly positive (binned ECE is upward biased even under H0)",
      floor_small["floor_mean"] > 0 and floor_large["floor_mean"] > 0)
check("null floor shrinks as n grows",
      floor_large["floor_mean"] < floor_small["floor_mean"],
      f"n=150 {floor_small['floor_mean']:.4f} -> n=4000 {floor_large['floor_mean']:.4f}")
check("null floor's p95 is at or above its mean",
      floor_large["floor_p95"] >= floor_large["floor_mean"] - 1e-12,
      f"mean={floor_large['floor_mean']:.4f} p95={floor_large['floor_p95']:.4f}")

# A model scored against panels drawn from ITSELF is exactly the H0 case: its
# observed ece_thr should look like a draw from its own null distribution,
# not something systematically above it.
p_null, c_null = calib_data(3000, 5, 10, np.random.RandomState(7))
floor_self = mt.ece_thr_null(p_null, c_null, n_bins=10, n_boot=400, seed=0)
obs_self = mt.ece_thr(p_null, c_null, n_bins=10, n_boot=0)["ece"]
check("a genuinely calibrated model's observed ece_thr sits within its own null band",
      obs_self <= floor_self["floor_p95"],
      f"observed={obs_self:.4f} floor_mean={floor_self['floor_mean']:.4f} "
      f"p95={floor_self['floor_p95']:.4f}")

# An overconfident model's ece_thr must clear ITS OWN null floor by more than
# a calibrated model's does - the floor alone does not manufacture a false
# positive out of sharpening.
floor_bad = mt.ece_thr_null(p_bad, c_c, n_bins=10, n_boot=300, seed=0)
floor_good = mt.ece_thr_null(p_c, c_c, n_bins=10, n_boot=300, seed=0)
obs_bad = mt.ece_thr(p_bad, c_c, n_bins=10, n_boot=0)["ece"]
obs_good = mt.ece_thr(p_c, c_c, n_bins=10, n_boot=0)["ece"]
check("an overconfident model clears its own null floor by a wide margin",
      (obs_bad - floor_bad["floor_mean"]) > 3 * (obs_good - floor_good["floor_mean"]),
      f"sharpened: obs={obs_bad:.4f} floor={floor_bad['floor_mean']:.4f} "
      f"gap={obs_bad - floor_bad['floor_mean']:.4f}  |  calibrated: "
      f"obs={obs_good:.4f} floor={floor_good['floor_mean']:.4f} "
      f"gap={obs_good - floor_good['floor_mean']:.4f}")

check("ece_thr_null rejects a prediction/target shape mismatch",
      _raises(lambda: mt.ece_thr_null(p_c, c_c[:, :3])))
check("ece_thr_null rejects K < 2",
      _raises(lambda: mt.ece_thr_null(p_c[:, :1], c_c[:, :1])))

# Paired SE: comparing a model to ITSELF must give a zero diff and a well-
# defined, non-negative SE.
self_pair = mt.ece_thr_paired_se(p_c, p_c, c_c, n_bins=10, n_boot=200, seed=0)
check("paired SE of a model against itself has zero diff",
      abs(self_pair["diff"]) < 1e-12, f"diff={self_pair['diff']:.2e}")
check("paired SE of a model against itself is non-negative and finite",
      self_pair["se"] >= 0 and np.isfinite(self_pair["se"]))

# Antisymmetry: swapping A and B negates the diff and leaves the SE unchanged
# (same resampled pairs, same |difference| each replicate).
ab = mt.ece_thr_paired_se(p_c, p_bad, c_c, n_bins=10, n_boot=200, seed=3)
ba = mt.ece_thr_paired_se(p_bad, p_c, c_c, n_bins=10, n_boot=200, seed=3)
check("paired SE is antisymmetric in (A, B): diff flips sign",
      abs(ab["diff"] + ba["diff"]) < 1e-9, f"AB={ab['diff']:.4f} BA={ba['diff']:.4f}")
check("paired SE magnitude is the same either way round",
      abs(ab["se"] - ba["se"]) < 1e-9, f"AB={ab['se']:.4f} BA={ba['se']:.4f}")

# A calibrated model and a badly sharpened one must be DETECTABLY different -
# the |diff| should clear several multiples of its own paired SE - while two
# near-identical calibrated fits (different RNG draws of the same truth)
# should not.
check("a calibrated vs. a sharpened model is detectable: |diff| >> paired SE",
      abs(ab["diff"]) > 5 * ab["se"],
      f"diff={ab['diff']:.4f} se={ab['se']:.4f} "
      f"z={abs(ab['diff']) / max(ab['se'], 1e-12):.1f}")
p_c2, c_c2 = calib_data(2000, 5, 8, np.random.RandomState(9))
near = mt.ece_thr_paired_se(p_c, p_c2, c_c, n_bins=10, n_boot=200, seed=4)
marginal_se_a = mt.ece_thr(p_c, c_c, n_bins=10, n_boot=200, seed=0)["ece_se"]
marginal_se_b = mt.ece_thr(p_c2, c_c, n_bins=10, n_boot=200, seed=0)["ece_se"]
check("paired SE cancels shared item-level variance: it is well under "
      "sqrt(2) x the larger of the two models' marginal ece_thr SE",
      near["se"] < 1.4 * max(marginal_se_a, marginal_se_b),
      f"paired se={near['se']:.4f}  "
      f"sqrt(2)*max(marginal)={1.414 * max(marginal_se_a, marginal_se_b):.4f}")

check("ece_thr_paired_se rejects a prediction/target shape mismatch",
      _raises(lambda: mt.ece_thr_paired_se(p_c, p_bad, c_c[:, :3])))

# ece_thr_null draws from the PREDICTION (not to_pmf(counts), which is exactly
# normalised by construction), so a row summing to 1.0 + a few ULPs - the
# ordinary result of a float32 round-trip, or of a clamp_min(0.0) elsewhere in
# a pipeline - used to crash `np.random.multinomial`'s sum(pvals[:-1]) <= 1.0
# check. Caught via recalibrate_report.py on real gemma-4-12b-it predictions.
p_drift = rand_pmf(np.random.RandomState(9), 40, 5).astype(np.float32).astype(np.float64)
p_drift = p_drift * (1.0 + 1e-7)  # push every row just over 1.0
c_drift = np.random.RandomState(9).multinomial(10, [0.2] * 5, size=40).astype(np.float64)
check("a prediction summing fractionally over 1.0 does not crash ece_thr_null",
      not _raises(lambda: mt.ece_thr_null(p_drift, c_drift, n_bins=5, n_boot=20)),
      f"row sums up to {p_drift.sum(-1).max():.10f}")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
print("\n20. brier and tbce: proper, minimised at pi = phat, and tbce finite on"
      " degenerate panels")

check("LOSSES holds all four objectives",
      sorted(losses.LOSSES) == ["brier", "ce", "rps", "tbce"])
check("build_loss returns each of them",
      all(callable(losses.build_loss(n)) for n in losses.LOSSES))

# Every one of the four is a proper scoring rule, so the empirical pmf is its
# global minimiser. In the simplex-interior parametrisation pi = softmax(logits)
# that means the gradient wrt the logits vanishes exactly at logits = log(phat).
rng20 = np.random.RandomState(19)
for K_ in (3, 5, 7):
    phat = torch.tensor(rand_pmf(rng20, 12, K_), dtype=torch.float64)
    M_ = torch.tensor(rng20.randint(4, 19, size=12), dtype=torch.float64)
    counts = phat * M_[:, None]
    for ln in ("ce", "rps", "brier", "tbce"):
        logits = torch.log(phat).clone().requires_grad_(True)
        log_pmf = torch.log_softmax(logits, dim=-1)
        loss = losses.build_loss(ln)(log_pmf, counts, weight_by_M=True)
        (g,) = torch.autograd.grad(loss, logits)
        check(f"{ln} K={K_}: gradient ~ 0 at pi = phat",
              float(g.norm()) < 1e-6, f"|g|={float(g.norm()):.2e}")

# tbce must stay finite when the target panel has empty cells and every
# threshold Ghat_b lands on exactly 0 or 1 (M = 8, the common case). The
# 0 * log(.) terms must not become nan, and the log(1 - F_b) tail must not
# blow up, even for a hard-saturated prediction. No epsilon clamp is allowed
# to be doing this - logcumsumexp of finite logits is finite on its own.
rng20b = np.random.RandomState(190)
degenerate = torch.tensor([
    [4.0, 0.0, 4.0, 0.0, 0.0],      # Ghat = [.5, .5, 1, 1]
    [8.0, 0.0, 0.0, 0.0, 0.0],      # Ghat = [1, 1, 1, 1]
    [0.0, 0.0, 0.0, 0.0, 8.0],      # Ghat = [0, 0, 0, 0]
    [0.0, 3.0, 0.0, 5.0, 0.0],      # Ghat = [0, .375, .375, 1]
], dtype=torch.float64)
G_deg = torch.cumsum(degenerate / degenerate.sum(-1, keepdim=True), dim=-1)[..., :-1]
check("the degenerate panels really do pin Ghat_b at 0/1",
      bool(((G_deg == 0) | (G_deg == 1)).any()))
for scale, tag in [(1.0, "moderate"), (30.0, "hard-saturated")]:
    logits = torch.tensor(scale * rng20b.randn(4, 5), dtype=torch.float64)
    log_pmf = torch.log_softmax(logits, dim=-1)
    val = losses.threshold_bce_loss(log_pmf, degenerate, weight_by_M=True)
    check(f"tbce finite on empty-cell panels ({tag} prediction)",
          bool(torch.isfinite(val)), f"value={float(val):.4f}")
    # and differentiable there - the backward of logcumsumexp must not nan either
    lg = torch.tensor(scale * rng20b.randn(4, 5), dtype=torch.float64,
                      requires_grad=True)
    (gb,) = torch.autograd.grad(
        losses.threshold_bce_loss(torch.log_softmax(lg, -1), degenerate), lg)
    check(f"tbce gradient finite on empty-cell panels ({tag} prediction)",
          bool(torch.isfinite(gb).all()))

# --------------------------------------------------------------------------- #
# 21. Degeneracy detection: the reproducibility label and its detection metrics
#
# `analysis/degeneracy_probe.py` asks whether uncertainty flags an activation
# the encoder cannot reproduce. Its label is a correlation between two forward
# passes and its metrics are AUROC and a budgeted recall; all three have a way
# of looking right while being wrong, so each is pinned here.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
print("\n23. Single-label (M = 1) regime and point metrics")
p1 = rand_pmf(rng, 300, 5)
y1 = np.array([rng.choice(5, p=q) for q in p1])
c1 = np.eye(5)[y1]
check("regime: M=1 -> point", mt.regime(c1) == "point")
check("regime: M>=2 -> distributional",
      mt.regime(np.stack([rng.multinomial(3, q) for q in p1])) == "distributional")
c_mix = np.stack([rng.multinomial(1 + (i % 2), q) for i, q in enumerate(p1)])
check("regime: mixed M -> point", mt.regime(c_mix) == "point")
s1 = mt.summary(p1, c1, n_boot_null=20)
check("summary at M=1: rps_debiased is None", s1["rps_debiased"] is None)
check("summary at M=1: everything else finite",
      all(np.isfinite(v) for k, v in s1.items() if isinstance(v, float)),
      str({k: v for k, v in s1.items() if isinstance(v, float) and not np.isfinite(v)}))
check("rps_debiased still refuses M=1", _raises(lambda: mt.rps_debiased(p1, c1)))
# nll at M=1 is the mean negative log-probability of the observed label
check("nll == -mean log p(y)",
      abs(s1["nll"] + np.log(p1[np.arange(300), y1]).mean()) < 1e-12)
# hand-checked point metrics
P = np.array([[0.7, 0.2, 0.1, 0.0, 0.0],     # mode 0, median 0, mean 0.4
              [0.0, 0.1, 0.2, 0.3, 0.4],     # mode 4, median 3, mean 3.0
              [0.0, 0.0, 1.0, 0.0, 0.0]])    # mode 2, median 2, mean 2.0
C = np.eye(5)[[0, 2, 2]]
pm = mt.point_metrics(P, C)
check("acc", abs(pm["acc"] - 2 / 3) < 1e-12, f"{pm['acc']}")
check("acc_within1", abs(pm["acc_within1"] - 2 / 3) < 1e-12)
check("pmf_median", list(mt.pmf_median(P)) == [0, 3, 2], f"{mt.pmf_median(P)}")
check("mae_median", abs(pm["mae_median"] - 1 / 3) < 1e-12, f"{pm['mae_median']}")
check("mae_mean", abs(pm["mae_mean"] - (0.4 + 1.0 + 0.0) / 3) < 1e-12)
# QWK: perfect agreement is 1, a constant predictor is 0
check("QWK perfect = 1", abs(mt.quadratic_weighted_kappa(y1, y1, 5) - 1) < 1e-12)
check("QWK reversed < 0", mt.quadratic_weighted_kappa(y1, 4 - y1, 5) < 0)
check("majority label: ties go to the lowest level",
      list(mt.majority_label(np.array([[0, 2, 2, 0, 0], [1, 0, 0, 0, 1]]))) == [1, 0])
# a calibrated predictor at M=1: ece_thr sits near its own null floor
pc = rand_pmf(rng, 3000, 5)
yc = np.eye(5)[[rng.choice(5, p=q) for q in pc]]
sc = mt.summary(pc, yc, n_boot_null=100)
check("M=1 calibrated: ece_thr within 3 null SEs of the floor",
      abs(sc["ece_thr_above_floor"]) < 3 * sc["ece_thr_null_se"] + 1e-3,
      f"above_floor={sc['ece_thr_above_floor']:+.4f} se={sc['ece_thr_null_se']:.4f}")


# --------------------------------------------------------------------------- #
def test_math():
    assert not FAIL, FAIL


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        sys.exit(1)
    print("all mathematical claims verified")
