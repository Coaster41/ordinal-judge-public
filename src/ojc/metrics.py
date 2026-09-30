"""Scoring rules and estimators for distributional ordinal targets.

Everything here is written against *counts* with a per-item annotator count
M_i, because the finite-M corrections below all depend on M_i and several of
them are wrong if you silently assume M is constant.

Key facts (derived and unit-tested in tests/test_math.py):

  * soft-label CE is LINEAR in the target  -> unbiased for the population pmf
  * RPS is QUADRATIC in the target         -> unbiased up to an additive
    constant Sum_k G_k(1-G_k)/M_i which does not depend on the prediction, so
    the minimiser is unchanged, but the constant varies across items once M
    varies and therefore must be removed before averaging (-> rps_debiased)
  * entropy is CONCAVE in the target       -> plug-in H[phat] is biased DOWN by
    about (K-1)/(2M) nats (Miller-Madow); this bias varies with M_i and so
    induces a spurious correlation between "aleatoric" and annotator count
  * EMD uses |.|, which is CONVEX          -> biased UP, and the bias depends on
    the prediction, so EMD can misrank. Reported, but secondary.

Single-label data (M = 1). `rps_debiased` and the Miller-Madow entropy need
M >= 2 and are undefined there; `summary` then reports them as None and the
POINT metrics of `point_metrics` (accuracy, MAE, QWK, Spearman, NLL) become the
headline. Everything else - raw RPS, soft CE (= NLL of the observed label),
`ece_top` (= classic top-label ECE on a 0/1 target) and `ece_thr` with its null
floor - is well defined at any M. `regime` names which case applies.
"""

from __future__ import annotations

import warnings

import numpy as np

EPS = 1e-12


def to_pmf(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    return counts / counts.sum(-1, keepdims=True)


def cdf(p: np.ndarray) -> np.ndarray:
    """CDF over the ordinal support, dropping the final (always 1.0) entry."""
    return np.cumsum(np.asarray(p, dtype=np.float64), axis=-1)[..., :-1]


# --------------------------------------------------------------------------- #
# Ranked probability score  (= discrete CRPS = Cramer distance on unit support)
# --------------------------------------------------------------------------- #

def rps(pred_pmf: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Raw per-item RPS against the empirical target. Sum_k (F_k - Ghat_k)^2."""
    F = cdf(pred_pmf)
    G = cdf(to_pmf(counts))
    return ((F - G) ** 2).sum(-1)


def rps_debiased(pred_pmf: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Unbiased estimator of the POPULATION RPS(F, G_true).

    E[RPS(F, Ghat)] = RPS(F, G) + Sum_k G_k(1-G_k)/M.
    Since E[Ghat_k(1-Ghat_k)] = G_k(1-G_k)(1 - 1/M), an unbiased estimate of
    that offset is Sum_k Ghat_k(1-Ghat_k)/(M-1). Requires M >= 2.

    With constant M this differs from `rps` by a constant and changes no
    ranking; with variable M (CommitmentBank: M in 8..18) it is the only
    version comparable across items.
    """
    counts = np.asarray(counts, dtype=np.float64)
    M = counts.sum(-1)
    if np.any(M < 2):
        raise ValueError("rps_debiased needs M >= 2 for every item")
    G = cdf(to_pmf(counts))
    offset = (G * (1.0 - G)).sum(-1) / (M - 1.0)
    return rps(pred_pmf, counts) - offset


# --------------------------------------------------------------------------- #
# Earth mover's distance (1-Wasserstein on unit-spaced ordinal support)
# --------------------------------------------------------------------------- #

def emd(pred_pmf: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Sum_k |F_k - Ghat_k|. Biased upward under finite M; see module docstring."""
    F = cdf(pred_pmf)
    G = cdf(to_pmf(counts))
    return np.abs(F - G).sum(-1)


def emd_noise_floor(counts: np.ndarray, n_rep: int = 200, seed: int = 0) -> float:
    """EMD between two disjoint halves of the annotator panel, averaged.

    This is the EMD a *perfect* model would still incur from finite-M noise, so
    it is the reference line for the raw EMD numbers.
    """
    rng = np.random.RandomState(seed)
    counts = np.asarray(counts, dtype=np.int64)
    out = []
    for _ in range(n_rep):
        a_all, b_all = [], []
        for c in counts:
            labels = np.repeat(np.arange(len(c)), c)
            labels = rng.permutation(labels)
            h = len(labels) // 2
            a = np.bincount(labels[:h], minlength=len(c))
            b = np.bincount(labels[h: 2 * h], minlength=len(c))
            a_all.append(a)
            b_all.append(b)
        out.append(emd(to_pmf(np.stack(a_all)), np.stack(b_all)).mean())
    return float(np.mean(out))


# --------------------------------------------------------------------------- #
# Cross entropy / NLL
# --------------------------------------------------------------------------- #

def soft_ce(pred_pmf: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-annotator soft-label cross entropy (nats)."""
    p = to_pmf(counts)
    return -(p * np.log(np.clip(pred_pmf, EPS, None))).sum(-1)


def multinomial_nll(pred_pmf: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Total multinomial NLL per item (up to the count-only normaliser).

    Equals M_i * soft_ce, which is exactly why the M-weighted CE is the MLE.
    """
    counts = np.asarray(counts, dtype=np.float64)
    return -(counts * np.log(np.clip(pred_pmf, EPS, None))).sum(-1)


# --------------------------------------------------------------------------- #
# Entropy and its finite-M correction
# --------------------------------------------------------------------------- #

def entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return -(p * np.log(np.clip(p, EPS, None))).sum(-1)


def entropy_miller_madow(counts: np.ndarray) -> np.ndarray:
    """Miller-Madow bias-corrected plug-in entropy: H_hat + (K_obs - 1)/(2M).

    K_obs is the number of levels actually observed, not K, which is the
    standard form and is much better behaved when many levels are empty.
    """
    counts = np.asarray(counts, dtype=np.float64)
    M = counts.sum(-1)
    k_obs = (counts > 0).sum(-1)
    return entropy(to_pmf(counts)) + (k_obs - 1.0) / (2.0 * M)


# --------------------------------------------------------------------------- #
# Miscellaneous
# --------------------------------------------------------------------------- #

def pmf_mean(pred_pmf: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
    p = np.asarray(pred_pmf, dtype=np.float64)
    v = np.arange(p.shape[-1]) if values is None else values
    return (p * v).sum(-1)


def coverage(pred_pmf: np.ndarray, counts: np.ndarray, level: float = 0.9) -> tuple[float, float]:
    """Annotator-level coverage of the equal-tailed central interval.

    An annotator is 'covered' if their score lies in the interval, so the
    target is the fraction of *annotations* (not items) inside it.
    """
    p = np.asarray(pred_pmf, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    C = np.cumsum(p, -1)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    lo = (C < lo_q).sum(-1)
    hi = (C < hi_q).sum(-1)
    hi = np.minimum(hi, p.shape[-1] - 1)
    idx = np.arange(p.shape[-1])[None, :]
    inside = (idx >= lo[:, None]) & (idx <= hi[:, None])
    cov = (counts * inside).sum() / counts.sum()
    width = (hi - lo + 1.0).mean()
    return float(cov), float(width)


# --------------------------------------------------------------------------- #
# Calibration:  ECE over {mode, full} x {pmf, cdf}
# --------------------------------------------------------------------------- #
#
# Both dimensions reduce to the same computation over (prediction, target)
# pairs, which is why they are one function rather than four:
#
#   dim 1 -- what the prediction is scored against, i.e. the TARGET matrix.
#     "full"  every annotator's rating counts separately, but each ITEM carries
#             total weight 1 regardless of panel size. Rather than expanding the
#             panel into M_i rows and down-weighting each by 1/M_i, we use the
#             empirical pmf c_i/M_i directly as the target - identical in
#             expectation and in every bin mean, since a bin mean of the
#             1/M-weighted indicators IS the mean of c_i/M_i. This matters:
#             panel size ranges from ~2.8 ratings/item (MHS) to 100 (ChaosNLI),
#             so unweighted pooling would let a handful of corpora own the curve.
#     "mode"  the panel is first collapsed to its majority level. Ties are split
#             evenly across the tied levels instead of being broken by index
#             order, which would otherwise bias every tied item toward level 0.
#
#   dim 2 -- which events are being calibrated, i.e. the SPACE.
#     "pmf"   the K events {y = k}. Ordering-blind; this is the usual multiclass
#             (classwise) ECE.
#     "cdf"   the K-1 events {y <= k}. The ordinal-aware version: a model that
#             puts its mass one level off is barely penalised in cdf space and
#             maximally penalised in pmf space. Reporting both is the point -
#             the gap between them is a statement about ordering.
#
# BIAS WARNING, and it is not a small one. The two modes do not merely estimate
# the same quantity with different variance - they are calibrated against
# DIFFERENT targets. The majority level of a panel is systematically sharper
# than the per-annotator distribution, so the model that minimises ece(mode=...)
# is the distribution of the majority statistic, not the distribution of the
# raters. A model that is perfectly calibrated per annotator therefore looks
# badly calibrated in "mode", and an OVERCONFIDENT model scores better:
# tests/test_math.py pins this at M=8, where sharpening a calibrated predictor
# to p^3 moves ECE(mode) from 0.078 down to 0.030 while ECE(full) correctly
# rises from 0.004 to 0.109.
#
# So: "full" is the one to read as "does this match the raters". "mode" answers
# the narrower question the calibration literature usually asks - given that a
# downstream consumer will act on the majority label, is the model's confidence
# in that label honest - and is reported for comparability, not because it is
# the better estimator of anything here.


def _target_matrix(counts: np.ndarray, mode: str) -> np.ndarray:
    """(n, K) target in probability units. See the note above on tie splitting."""
    counts = np.asarray(counts, dtype=np.float64)
    if mode == "full":
        return to_pmf(counts)
    if mode == "mode":
        top = counts.max(-1, keepdims=True)
        win = (counts >= top - EPS).astype(np.float64)
        return win / win.sum(-1, keepdims=True)
    raise ValueError(f"mode must be 'mode' or 'full', got {mode!r}")


def _bin_edges(pred: np.ndarray, n_bins: int, binning: str) -> np.ndarray:
    if binning == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if binning == "quantile":
        # equal-mass bins. With pmf-space predictions most probabilities sit
        # near 0, so uniform bins can leave the top half of the range empty;
        # quantile bins spend resolution where the predictions actually are.
        q = np.quantile(pred.ravel(), np.linspace(0.0, 1.0, n_bins + 1))
        q[0], q[-1] = 0.0, 1.0
        return np.unique(q)
    raise ValueError(f"binning must be 'uniform' or 'quantile', got {binning!r}")


def _curve(binidx: np.ndarray, pred: np.ndarray, targ: np.ndarray, nb: int):
    """Weighted per-bin (count, mean prediction, mean target) and the ECE."""
    w = np.bincount(binidx, minlength=nb).astype(np.float64)
    sp = np.bincount(binidx, weights=pred, minlength=nb)
    st = np.bincount(binidx, weights=targ, minlength=nb)
    nz = w > 0
    conf = np.full(nb, np.nan)
    acc = np.full(nb, np.nan)
    conf[nz] = sp[nz] / w[nz]
    acc[nz] = st[nz] / w[nz]
    ece = float((w[nz] * np.abs(conf[nz] - acc[nz])).sum() / w.sum()) if w.sum() else np.nan
    return w, conf, acc, ece


def ece(pred_pmf: np.ndarray, counts: np.ndarray, mode: str = "full",
        space: str = "pmf", n_bins: int = 15, binning: str = "uniform",
        n_boot: int = 200, seed: int = 0, level: float = 0.95) -> dict:
    """Expected calibration error plus the reliability curve with error bars.

    Returns the scalar `ece`, the worst single bin `mce`, and a `curve` block
    holding per-bin weight, mean confidence, observed frequency and a bootstrap
    interval on that frequency.

    ERROR BARS. The interval is a two-level bootstrap: items are resampled with
    replacement, and within each drawn item its panel is resampled with
    replacement too. Both levels are needed and neither substitutes for the
    other. Resampling items alone would ignore that the observed frequency in a
    bin is itself an average of finite panels - the dominant term for a 3-rater
    corpus - while resampling raters alone would treat this particular test set
    as the population. Resampling ITEMS (not (item, level) pairs) is also what
    keeps the K events of one item together: they are strongly dependent, and
    breaking them apart would give intervals that are far too tight.

    Bin membership is a property of the prediction, which no resampling touches,
    so the bins are fixed across replicates and the intervals are comparable
    bin by bin.

    n_boot=0 skips the bootstrap and returns the point estimates only.
    """
    p = np.asarray(pred_pmf, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if p.shape != counts.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs counts {counts.shape}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    t = _target_matrix(counts, mode)
    if space == "cdf":
        p, t = cdf(p), cdf(t)
    elif space != "pmf":
        raise ValueError(f"space must be 'pmf' or 'cdf', got {space!r}")

    n, C = p.shape
    edges = _bin_edges(p, n_bins, binning)
    nb = len(edges) - 1
    # right-closed except for the first bin, so 0.0 and 1.0 both land in-range
    binidx = np.clip(np.searchsorted(edges, p, side="left") - 1, 0, nb - 1)

    w, conf, acc, e = _curve(binidx.ravel(), p.ravel(), t.ravel(), nb)
    nz = w > 0
    gap = np.abs(conf - acc)
    out = {
        "ece": e,
        "mce": float(np.nanmax(gap)) if nz.any() else np.nan,
        "mode": mode, "space": space, "n_bins": nb, "binning": binning,
        "n_items": int(n), "n_events": int(n * C),
        "curve": {
            "edges": edges.tolist(),
            "weight": w.tolist(),
            "frac": (w / w.sum()).tolist(),
            "conf": conf.tolist(),
            "acc": acc.tolist(),
            "gap": gap.tolist(),
        },
    }
    if n_boot <= 0:
        return out

    rng = np.random.RandomState(seed)
    M = counts.sum(-1)
    phat = to_pmf(counts)
    accs = np.full((n_boot, nb), np.nan)
    eces = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        cb = np.stack([rng.multinomial(int(M[i]), phat[i]) for i in idx]).astype(np.float64)
        tb = _target_matrix(cb, mode)
        if space == "cdf":
            tb = cdf(tb)
        _, _, accs[b], eces[b] = _curve(binidx[idx].ravel(), p[idx].ravel(),
                                        tb.ravel(), nb)
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    # bins that no prediction ever falls into stay NaN in every replicate; that
    # is the correct answer for them, not a numerical problem worth a warning
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["curve"]["acc_lo"] = np.nanpercentile(accs, lo_q, axis=0).tolist()
        out["curve"]["acc_hi"] = np.nanpercentile(accs, hi_q, axis=0).tolist()
        out["curve"]["acc_se"] = np.nanstd(accs, axis=0, ddof=1).tolist()
    out["ece_se"] = float(np.std(eces, ddof=1))
    out["ece_lo"] = float(np.percentile(eces, lo_q))
    out["ece_hi"] = float(np.percentile(eces, hi_q))
    out["n_boot"] = int(n_boot)
    out["level"] = float(level)
    return out


ECE_GRID = [(m, s) for m in ("mode", "full") for s in ("pmf", "cdf")]


def ece_grid(pred_pmf: np.ndarray, counts: np.ndarray, n_bins: int = 15,
             binning: str = "uniform", n_boot: int = 200, seed: int = 0) -> dict:
    """All four {mode, full} x {pmf, cdf} cells, keyed 'ece_<mode>_<space>'."""
    return {f"ece_{m}_{s}": ece(pred_pmf, counts, mode=m, space=s, n_bins=n_bins,
                                binning=binning, n_boot=n_boot, seed=seed)
            for m, s in ECE_GRID}


# --------------------------------------------------------------------------- #
# Calibration:  top-label ECE, and threshold-averaged CDF ECE
# --------------------------------------------------------------------------- #
#
# Neither of these is one of the four ece_grid cells above, despite looking
# close to two of them:
#
#   ece_top vs ece(mode="mode", space="pmf")
#     ece(mode="mode") pools ALL K classes' predicted probabilities into one
#     set of bins (a classwise/static multiclass ECE) and targets the
#     majority-vote one-hot panel. ece_top uses exactly ONE number per item -
#     the model's confidence in the class it actually emits, argmax_k phat_k -
#     binned and compared against the fraction of the ACTUAL panel (not the
#     majority-collapsed one) that agrees with that one emitted class. This is
#     the textbook top-label ECE (Guo et al. 2017), generalised to a soft
#     (per-annotator) target the way "full" mode is elsewhere in this file.
#
#   ece_thr vs ece(mode="full", space="cdf")
#     ece(mode="full", space="cdf") pools the K-1 threshold events {y<=j} for
#     EVERY j together into one set of bins before computing a single ECE -
#     so an item's cut point j=1 and its cut point j=3 can land in the same
#     bin and have their conf/acc AVERAGED TOGETHER before the gap is taken.
#     ece_thr bins each cut point j SEPARATELY (its own bin edges, from its
#     own P_j values), computes ECE_j for that cut point alone, and only then
#     averages the K-1 per-threshold ECE_j with equal weight 1/(K-1). Because
#     |mean(x) - mean(y)| <= mean(|x-y|), pooling first can only average away
#     miscalibration that cancels across different cut points landing in the
#     same bin; ece_thr cannot hide that cancellation. The two are generally
#     different numbers, not two names for the same computation.


def ece_top(pred_pmf: np.ndarray, counts: np.ndarray, n_bins: int = 15,
            binning: str = "uniform", n_boot: int = 200, seed: int = 0,
            level: float = 0.95) -> dict:
    """Top-label calibration error.

        ECE_top = Sum_b (n_b/N) |mean_b(conf) - mean_b(agree)|

    where conf_i = max_k phat_k(i), yhat_i = argmax_k phat_k(i), and
    agree_i = counts[i, yhat_i] / M_i is the fraction of item i's OWN panel
    that rated it at the model's emitted category (the soft-label analogue of
    the hard indicator 1[yhat_i = y_i]). Items are binned by conf_i alone -
    one number per item, unlike ece_grid's per-(item,class) pooling.

    Same two-level bootstrap as `ece`: items resampled, then each resampled
    item's panel redrawn from Multinomial(M_i, phat_i). yhat_i and its bin are
    a property of the (unresampled) prediction and are held fixed.
    """
    p = np.asarray(pred_pmf, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if p.shape != counts.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs counts {counts.shape}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    n = p.shape[0]
    rows = np.arange(n)
    yhat = p.argmax(-1)
    conf = p[rows, yhat]
    M = counts.sum(-1)
    agree = counts[rows, yhat] / M

    edges = _bin_edges(conf, n_bins, binning)
    nb = len(edges) - 1
    binidx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, nb - 1)

    w, cf, ac, e = _curve(binidx, conf, agree, nb)
    gap = np.abs(cf - ac)
    out = {
        "ece": e,
        "mce": float(np.nanmax(gap)) if (w > 0).any() else np.nan,
        "n_bins": nb, "binning": binning, "n_items": int(n),
        "curve": {
            "edges": edges.tolist(), "weight": w.tolist(),
            "frac": (w / w.sum()).tolist(),
            "conf": cf.tolist(), "acc": ac.tolist(), "gap": gap.tolist(),
        },
    }
    if n_boot <= 0:
        return out

    rng = np.random.RandomState(seed)
    phat = to_pmf(counts)
    accs = np.full((n_boot, nb), np.nan)
    eces = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        cb = np.stack([rng.multinomial(int(M[i]), phat[i]) for i in idx]).astype(np.float64)
        agree_b = cb[rows, yhat[idx]] / cb.sum(-1)
        _, _, accs[b], eces[b] = _curve(binidx[idx], conf[idx], agree_b, nb)
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["curve"]["acc_lo"] = np.nanpercentile(accs, lo_q, axis=0).tolist()
        out["curve"]["acc_hi"] = np.nanpercentile(accs, hi_q, axis=0).tolist()
        out["curve"]["acc_se"] = np.nanstd(accs, axis=0, ddof=1).tolist()
    out["ece_se"] = float(np.std(eces, ddof=1))
    out["ece_lo"] = float(np.percentile(eces, lo_q))
    out["ece_hi"] = float(np.percentile(eces, hi_q))
    out["n_boot"] = int(n_boot)
    out["level"] = float(level)
    return out


def ece_thr(pred_pmf: np.ndarray, counts: np.ndarray, n_bins: int = 15,
            binning: str = "uniform", n_boot: int = 200, seed: int = 0,
            level: float = 0.95) -> dict:
    """Threshold/CDF calibration error, averaged over cut points binned
    SEPARATELY per cut point (see the module note above for why this is not
    `ece(mode="full", space="cdf")`).

        ECE_thr = 1/(K-1) Sum_{j=1}^{K-1} Sum_b (n_jb/N) |mean_jb(P_j) - mean_jb(Fhat_j)|

    where P_j(i) = Sum_{k<=j} phat_k(i) is the predicted CDF at cut point j
    and Fhat_j(i) = counts[i, :j+1].sum()/M_i the empirical one. Returns the
    aggregate `ece` (the mean above) plus a `per_threshold` list with each
    cut point's own ece/curve, for plotting.
    """
    p = np.asarray(pred_pmf, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if p.shape != counts.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs counts {counts.shape}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    n, K = p.shape
    if K < 2:
        raise ValueError(f"ece_thr needs K >= 2 ordinal categories, got K={K}")
    n_thr = K - 1
    P = cdf(p)                    # (n, n_thr)
    Fhat = cdf(to_pmf(counts))    # (n, n_thr)

    per_thr = []
    for j in range(n_thr):
        edges = _bin_edges(P[:, j], n_bins, binning)
        nb = len(edges) - 1
        binidx = np.clip(np.searchsorted(edges, P[:, j], side="left") - 1, 0, nb - 1)
        w, cf, ac, e = _curve(binidx, P[:, j], Fhat[:, j], nb)
        resid = cf - ac
        gap = np.abs(resid)
        per_thr.append({"j": j, "edges": edges, "nb": nb, "binidx": binidx,
                        "w": w, "conf": cf, "acc": ac, "resid": resid,
                        "gap": gap, "ece": e})

    ece_val = float(np.mean([c["ece"] for c in per_thr]))
    mce_val = float(np.nanmax([np.nanmax(c["gap"]) if (c["w"] > 0).any() else np.nan
                               for c in per_thr]))
    out = {
        "ece": ece_val, "mce": mce_val,
        "n_bins": n_bins, "binning": binning, "n_items": int(n), "n_thresholds": n_thr,
        "per_threshold": [
            {"j": c["j"], "ece": c["ece"],
             "curve": {"edges": c["edges"].tolist(), "weight": c["w"].tolist(),
                      "frac": (c["w"] / c["w"].sum()).tolist(),
                      "conf": c["conf"].tolist(), "acc": c["acc"].tolist(),
                      "resid": c["resid"].tolist(), "gap": c["gap"].tolist()}}
            for c in per_thr
        ],
    }
    if n_boot <= 0:
        return out

    rng = np.random.RandomState(seed)
    M = counts.sum(-1)
    phat = to_pmf(counts)
    accs_j = [np.full((n_boot, c["nb"]), np.nan) for c in per_thr]
    eces = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        cb = np.stack([rng.multinomial(int(M[i]), phat[i]) for i in idx]).astype(np.float64)
        Fb = cdf(to_pmf(cb))
        thr_eces = np.empty(n_thr)
        for j, c in enumerate(per_thr):
            _, _, accs_j[j][b], thr_eces[j] = _curve(
                c["binidx"][idx], P[idx, j], Fb[:, j], c["nb"])
        eces[b] = thr_eces.mean()
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j in range(n_thr):
            out["per_threshold"][j]["curve"]["acc_lo"] = np.nanpercentile(
                accs_j[j], lo_q, axis=0).tolist()
            out["per_threshold"][j]["curve"]["acc_hi"] = np.nanpercentile(
                accs_j[j], hi_q, axis=0).tolist()
            out["per_threshold"][j]["curve"]["acc_se"] = np.nanstd(
                accs_j[j], axis=0, ddof=1).tolist()
    out["ece_se"] = float(np.std(eces, ddof=1))
    out["ece_lo"] = float(np.percentile(eces, lo_q))
    out["ece_hi"] = float(np.percentile(eces, hi_q))
    out["n_boot"] = int(n_boot)
    out["level"] = float(level)
    return out


def ece_thr_null(pred_pmf: np.ndarray, counts: np.ndarray, n_bins: int = 15,
                 binning: str = "uniform", n_boot: int = 500, seed: int = 0,
                 level: float = 0.95) -> dict:
    """Null floor for `ece_thr`: the distribution ECE_thr would take if the
    model's OWN predicted pmf were the true per-item annotator distribution.

    Binned ECE is biased upward - a perfectly calibrated model still scores
    above zero, by an amount that depends on bin occupancy and on each
    item's M_i - so an observed ece_thr is only evidence of miscalibration
    once it clears this floor, not zero.

    Parametric bootstrap under H0 (pred_pmf IS truth): for each of n_boot
    replicates, redraw every item's panel from Multinomial(M_i, pred_pmf(i))
    - never from the empirical counts, which is what `ece_thr`'s own
    bootstrap draws from and answers a different question (sampling
    uncertainty in the ESTIMATE, not the floor a true model would show).
    P_j and its bins are a property of pred_pmf alone and do not change
    across replicates; only the redrawn O_j does - same binning procedure,
    same bin edges, as the real computation.

    Returns `floor_mean` (report as the floor) and `floor_p<100*level>` (its
    upper quantile), plus the full `eces` distribution for shading a plot.
    """
    p = np.asarray(pred_pmf, dtype=np.float64)
    # `np.random.multinomial` requires sum(pvals[:-1]) <= 1.0 EXACTLY - unlike
    # every other bootstrap in this file, which redraws from `to_pmf(counts)`
    # (exact by construction), this one redraws from the PREDICTION, which can
    # carry float32-rounding or clamp-induced drift of a few ULPs above 1.0 and
    # crash the draw. Renormalising is a no-op for an already-valid pmf and a
    # fix for everything else.
    p = p / p.sum(-1, keepdims=True)
    counts = np.asarray(counts, dtype=np.float64)
    if p.shape != counts.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs counts {counts.shape}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    n, K = p.shape
    if K < 2:
        raise ValueError(f"ece_thr_null needs K >= 2 ordinal categories, got K={K}")
    n_thr = K - 1
    P = cdf(p)
    M = counts.sum(-1)

    per_thr_bins = []
    for j in range(n_thr):
        edges = _bin_edges(P[:, j], n_bins, binning)
        nb = len(edges) - 1
        binidx = np.clip(np.searchsorted(edges, P[:, j], side="left") - 1, 0, nb - 1)
        per_thr_bins.append((nb, binidx))

    rng = np.random.RandomState(seed)
    eces = np.empty(n_boot)
    for b in range(n_boot):
        cb = np.stack([rng.multinomial(int(M[i]), p[i]) for i in range(n)]).astype(np.float64)
        Ob = cdf(to_pmf(cb))
        thr_eces = np.empty(n_thr)
        for j, (nb, binidx) in enumerate(per_thr_bins):
            _, _, _, thr_eces[j] = _curve(binidx, P[:, j], Ob[:, j], nb)
        eces[b] = thr_eces.mean()

    q = 100 * level
    return {
        "floor_mean": float(eces.mean()),
        f"floor_p{int(round(q))}": float(np.percentile(eces, q)),
        "floor_se": float(np.std(eces, ddof=1)),
        "n_boot": int(n_boot), "level": float(level), "n_items": int(n),
        "eces": eces.tolist(),
    }


def ece_thr_paired_se(pred_pmf_a: np.ndarray, pred_pmf_b: np.ndarray,
                      counts: np.ndarray, n_bins: int = 15, binning: str = "uniform",
                      n_boot: int = 500, seed: int = 0) -> dict:
    """Paired cluster-bootstrap SE on ece_thr(A) - ece_thr(B), for two models
    scored on the SAME items.

    A and B are typically two backbones or two arms evaluated on identical
    (item, panel) pairs, so their marginal ECE_thr estimates share item-level
    variance; comparing marginal CIs is therefore overly conservative.
    Resamples ITEMS with replacement - never individual annotators, who are
    not independent draws within an item since they all share that item's
    P_j(i) - and recomputes ece_thr for BOTH models on the SAME resample, so
    the shared item-level variance cancels out of the stored difference. The
    SD of those differences is the SE on the gap, typically well under
    sqrt(2) times either model's marginal ece_thr SE.
    """
    Pa = np.asarray(pred_pmf_a, dtype=np.float64)
    Pb = np.asarray(pred_pmf_b, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if Pa.shape != counts.shape or Pb.shape != counts.shape:
        raise ValueError(f"shape mismatch: A {Pa.shape}, B {Pb.shape}, counts {counts.shape}")
    n = Pa.shape[0]

    ece_a = ece_thr(Pa, counts, n_bins=n_bins, binning=binning, n_boot=0)["ece"]
    ece_b = ece_thr(Pb, counts, n_bins=n_bins, binning=binning, n_boot=0)["ece"]

    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        ea = ece_thr(Pa[idx], counts[idx], n_bins=n_bins, binning=binning, n_boot=0)["ece"]
        eb = ece_thr(Pb[idx], counts[idx], n_bins=n_bins, binning=binning, n_boot=0)["ece"]
        diffs[b] = ea - eb

    return {
        "ece_thr_a": ece_a, "ece_thr_b": ece_b, "diff": ece_a - ece_b,
        "se": float(np.std(diffs, ddof=1)), "n_boot": int(n_boot), "n_items": int(n),
        "diffs": diffs.tolist(),
    }


def regime(counts: np.ndarray) -> str:
    """'distributional' when every item has M >= 2 (the finite-M corrections are
    defined), else 'point'. Mixed panels - some items with a single rating - are
    'point', because the distributional headline would silently drop items."""
    M = np.asarray(counts, dtype=np.float64).sum(-1)
    return "distributional" if M.min() >= 2 else "point"


def majority_label(counts: np.ndarray) -> np.ndarray:
    """Plurality level per item. Ties go to the LOWEST tied level (np.argmax);
    at M = 1 this is just the observed label."""
    return np.asarray(counts).argmax(-1)


def pmf_median(pred_pmf: np.ndarray) -> np.ndarray:
    """Smallest level k with F(k) >= 1/2 - the Bayes point prediction under
    absolute error."""
    F = cdf(pred_pmf)
    return (F < 0.5 - 1e-12).sum(-1)


def quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray, K: int) -> float:
    """Cohen's kappa with quadratic disagreement weights, for two integer label
    vectors on 0..K-1. 1 = perfect agreement, 0 = chance."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    O = np.zeros((K, K))
    np.add.at(O, (a, b), 1.0)
    i = np.arange(K)
    W = (i[:, None] - i[None, :]) ** 2 / max((K - 1) ** 2, 1)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1.0 - (W * O).sum() / den) if den > 0 else float("nan")


def point_metrics(pred_pmf: np.ndarray, counts: np.ndarray) -> dict:
    """Point-prediction metrics, the headline when M = 1 (see `regime`).

    With a panel (M > 1) the target point is the panel's plurality level (for
    accuracy / QWK) or its mean rating (for MAE / Spearman), so the numbers stay
    defined at any M - but at M >= 2 read the distributional metrics first.

      acc          mode of the pmf == plurality label
      acc_within1  |mode - plurality label| <= 1
      mae_median   |median of the pmf - mean rating|  (optimal under abs. error)
      mae_mean     |mean of the pmf - mean rating|
      rmse_mean    root mean squared error of the pmf mean
      qwk          quadratic-weighted kappa, pmf mode vs plurality label
      spearman     rank correlation of the pmf mean with the mean rating
      nll          mean negative log-probability of the observed ratings
    """
    from scipy.stats import spearmanr

    p = np.asarray(pred_pmf, dtype=np.float64)
    K = p.shape[-1]
    y_maj = majority_label(counts)
    y_mean = pmf_mean(to_pmf(counts))
    p_mode = p.argmax(-1)
    p_mean = pmf_mean(p)
    p_med = pmf_median(p)
    rho = spearmanr(p_mean, y_mean).statistic if np.ptp(p_mean) > 0 and np.ptp(y_mean) > 0 else float("nan")
    return {
        "acc": float((p_mode == y_maj).mean()),
        "acc_within1": float((np.abs(p_mode - y_maj) <= 1).mean()),
        "mae_median": float(np.abs(p_med - y_mean).mean()),
        "mae_mean": float(np.abs(p_mean - y_mean).mean()),
        "rmse_mean": float(np.sqrt(((p_mean - y_mean) ** 2).mean())),
        "qwk": quadratic_weighted_kappa(p_mode, y_maj, K),
        "spearman": float(rho),
        "nll": float(soft_ce(p, counts).mean()),
    }


def summary(pred_pmf: np.ndarray, counts: np.ndarray, n_bins: int = 15,
           n_boot_null: int = 200, seed: int = 0) -> dict:
    """Scalar metrics for one set of predictions.

    Always: raw `rps`, `emd`, `soft_ce`, the 90% interval coverage/width, the
    two calibration errors `ece_top` and `ece_thr`, and every `point_metrics`
    entry. `rps_debiased` needs M >= 2 for every item and is None otherwise;
    `regime` says which case applies.

    Calibration is `ece_top` and `ece_thr` only, both as point estimates
    (n_boot=0). Call them directly for curves and error bars, and
    `ece_thr_paired_se` for a cross-model comparison.

    `ece_thr` always carries its null floor (`ece_thr_null_*`): a raw ece_thr
    is not interpretable on its own, because binned ECE is upward-biased even
    for a perfectly calibrated model (see `ece_thr_null`), so
    `ece_thr_above_floor` is the number to read. `n_boot_null=200` keeps the
    cost bounded; pass `n_boot_null=0` to skip it (e.g. inside a tight loop).
    """
    reg = regime(counts)
    cov90, w90 = coverage(pred_pmf, counts, 0.9)
    ece_thr_val = ece_thr(pred_pmf, counts, n_bins=n_bins, n_boot=0)["ece"]
    out = {
        "regime": reg,
        "ece_top": ece_top(pred_pmf, counts, n_bins=n_bins, n_boot=0)["ece"],
        "ece_thr": ece_thr_val,
        "rps": float(rps(pred_pmf, counts).mean()),
        "rps_debiased": (float(rps_debiased(pred_pmf, counts).mean())
                         if reg == "distributional" else None),
        "emd": float(emd(pred_pmf, counts).mean()),
        "soft_ce": float(soft_ce(pred_pmf, counts).mean()),
        "cov90": cov90,
        "width90": w90,
    }
    out.update(point_metrics(pred_pmf, counts))
    if n_boot_null and n_boot_null > 0:
        null = ece_thr_null(pred_pmf, counts, n_bins=n_bins, n_boot=n_boot_null, seed=seed)
        level_key = f"floor_p{int(round(100 * null['level']))}"
        out["ece_thr_null_mean"] = null["floor_mean"]
        out["ece_thr_null_se"] = null["floor_se"]
        out[f"ece_thr_null_{level_key[len('floor_'):]}"] = null[level_key]
        out["ece_thr_above_floor"] = ece_thr_val - null["floor_mean"]
    return out
