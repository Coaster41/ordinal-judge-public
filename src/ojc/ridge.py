"""Ridge regression on the raw activations, and the pmf baselines built from it.

These are the rungs below the trained heads: they get the SAME frozen
activations and the SAME splits, and differ only in how much structure they are
allowed to express.

  climatology   the training marginal, repeated for every test item. Uses no
                features at all, so it is the floor any feature-based arm has to
                clear.
  spike(alpha)  ridge predicts the panel MEAN, that mean is rounded to the
                nearest level, and alpha of the mass is placed there with the
                remaining 1-alpha spread uniformly over the other K-1 levels.
                Two variants: alpha fixed at 0.9, and alpha tuned on validation.

The spike family is the point of comparison for the heads. It is a *scalar*
regression dressed up as a distribution: it knows where the panel sits but has
exactly one global knob for how spread out the panel is. So the gap between
spike(alpha*) and a trained head is the value of predicting DISPERSION per item,
separated from the value of predicting location - which is the question the
whole project is about.

Weighting. The ridge is fit with sample weights w_i = M_i, matching the
M-weighting of the head losses. This is not cosmetic: Var(panel mean) =
sigma^2 / M_i, so weighting by M is the GLS solution, and on d3code, where M
ranges from 9 to 196, the unweighted fit would treat a 9-rater item and a
196-rater item as equally informative about the same regression surface.

Solver. The ridge path is solved exactly, in closed form, for every lambda at
once, by eigendecomposing whichever Gram matrix is smaller - X^T X when n >= d,
X X^T when n < d. Both give the identical coefficients (they are the primal and
dual forms of the same problem); picking the smaller one is what makes a
1,200-item corpus at 5,120 features cost the same as a 30,000-item one. There is
no iterative solver here and no tuning of one.
"""

from __future__ import annotations

import numpy as np
import torch

from ojc import metrics as mt

LAMBDAS = np.logspace(-2, 6, 33)
ALPHAS = np.linspace(0.05, 0.99, 95)


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

def standardizer(Xtr: torch.Tensor):
    """Centre and scale on TRAIN ONLY; returns the transform and its stats.

    There is no PCA: the L2 penalty handles d >> n, but the features still have
    to be put on a common scale or a single high-variance coordinate would
    absorb the penalty budget on its own.
    """
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    return (lambda X: (X - mu) / sd), {"mu": mu, "sd": sd}


# --------------------------------------------------------------------------- #
# weighted ridge path
# --------------------------------------------------------------------------- #

def ridge_path(X: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
               lambdas: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted ridge with an UNPENALISED intercept, solved for every lambda.

    minimise  sum_i w_i (y_i - b - x_i.beta)^2 + lambda ||beta||^2

    Returns (B, b) with B of shape (n_lambda, d) and b of shape (n_lambda,).
    The intercept is handled by weighted-centring rather than by augmenting X,
    which is what keeps it out of the penalty.
    """
    X = X.double()
    y = y.double()
    w = w.double()
    sw = w.sum()
    xbar = (w[:, None] * X).sum(0, keepdim=True) / sw
    ybar = float((w * y).sum() / sw)
    A = torch.sqrt(w)[:, None] * (X - xbar)
    r = torch.sqrt(w) * (y - ybar)
    n, d = A.shape
    lam = torch.as_tensor(np.asarray(lambdas), dtype=torch.float64, device=X.device)

    if n >= d:
        G = A.T @ A                                  # (d, d)
        ev, V = torch.linalg.eigh(G)
        z = V.T @ (A.T @ r)                          # (d,)
        coef = z[None, :] / (ev[None, :] + lam[:, None])
        B = coef @ V.T                               # (L, d)
    else:
        G = A @ A.T                                  # (n, n)
        ev, U = torch.linalg.eigh(G)
        z = U.T @ r                                  # (n,)
        coef = z[None, :] / (ev[None, :] + lam[:, None])
        B = (coef @ U.T) @ A                         # (L, d)

    b = ybar - B @ xbar.reshape(-1)
    return B, b


def predict(B: torch.Tensor, b: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """(n_lambda, n) predicted means - lambda on the rows, items on the columns."""
    return B @ X.double().T + b[:, None]


# --------------------------------------------------------------------------- #
# pmf baselines
# --------------------------------------------------------------------------- #

def spike(mean_pred: np.ndarray, K: int, alpha: float) -> np.ndarray:
    """alpha on the level nearest the predicted mean, (1-alpha)/(K-1) elsewhere."""
    idx = np.clip(np.rint(np.asarray(mean_pred)).astype(int), 0, K - 1)
    P = np.full((len(idx), K), (1.0 - alpha) / (K - 1))
    P[np.arange(len(idx)), idx] = alpha
    return P


def climatology(counts_tr: np.ndarray, n: int, pooled: bool = True) -> np.ndarray:
    """Training marginal, tiled over n items.

    `pooled` sums raw counts, so every ANNOTATION carries equal weight - the
    convention the M-weighted losses use, and the one that makes climatology the
    honest zero-feature competitor to an M-weighted head. The item-averaged
    alternative is returned separately by `climatology_both` for the record.
    """
    counts_tr = np.asarray(counts_tr, dtype=np.float64)
    if pooled:
        p = counts_tr.sum(0) / counts_tr.sum()
    else:
        p = mt.to_pmf(counts_tr).mean(0)
    return np.tile(p, (n, 1))


def climatology_both(counts_tr: np.ndarray, n: int) -> dict:
    return {"climatology": climatology(counts_tr, n, pooled=True),
            "climatology_itemavg": climatology(counts_tr, n, pooled=False)}


def tune_alpha(mean_va: np.ndarray, counts_va: np.ndarray, K: int,
               alphas: np.ndarray = ALPHAS) -> tuple[float, np.ndarray]:
    """alpha* minimising the validation RPS.

    Tuned on RAW RPS, which selects exactly the same alpha as the debiased RPS
    would: the debiasing offset Sum_k G(1-G)/(M_i-1) is a per-item constant that
    does not depend on the prediction, so it shifts every alpha's mean score by
    the same amount whatever the panel sizes. Raw RPS is also defined at M = 1,
    where the debiased one is not. `tests/test_heads.py` pins the equality.
    """
    scores = np.array([mt.rps(spike(mean_va, K, float(a)), counts_va).mean()
                       for a in alphas])
    return float(alphas[int(np.argmin(scores))]), scores


def fit_mean_ridge(Xtr, ytr, wtr, Xva, yva, wva, lambdas=LAMBDAS) -> dict:
    """Select lambda on the M-weighted validation MSE - the fit's own criterion.

    Selecting on the criterion being optimised, rather than on the downstream
    RPS of the spike, keeps this a clean 'best linear estimate of the panel
    mean'. The spike's own free parameter (alpha) is tuned separately, also on
    validation, so location and dispersion are each chosen on held-out data and
    neither is chosen on test.
    """
    B, b = ridge_path(Xtr, ytr, wtr, lambdas)
    pv = predict(B, b, Xva)                                    # (L, n_va)
    err = (pv - yva.double()[None, :]) ** 2
    wmse = (err * wva.double()[None, :]).sum(1) / wva.sum()
    j = int(torch.argmin(wmse))
    return {
        "B": B[j], "b": float(b[j]), "lam": float(lambdas[j]), "lam_index": j,
        "val_wmse": float(wmse[j]),
        "val_mae": float((pv[j] - yva.double()).abs().mean()),
        "path_val_wmse": [float(v) for v in wmse],
        "lambdas": [float(v) for v in lambdas],
        "at_grid_edge": j in (0, len(lambdas) - 1),
    }
