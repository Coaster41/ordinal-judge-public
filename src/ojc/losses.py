"""Training objectives on distributional ordinal targets.

Four losses - `soft_ce_loss` ("ce"), `rps_loss` ("rps"), `brier_pmf_loss`
("brier") and `threshold_bce_loss` ("tbce") - all take integer COUNTS, not a
normalised pmf, all share the signature (log_pmf, counts, weight_by_M=True) ->
scalar, and all reduce as (w * per_item).sum() / w.sum().

For cross entropy the weighting is not a heuristic - it is exactly the
multinomial log-likelihood:

    log L = Sum_i Sum_k c_ik log pi_k(x_i)
          = Sum_i M_i * [ Sum_k phat_ik log pi_k(x_i) ]

so `w_i = M_i` recovers the MLE. Equal weighting is a quasi-likelihood that
discards the precision information and over-weights thinly annotated items.

For RPS the same weighting is the GLS analogue, since Var(Ghat_ik) =
G_ik(1-G_ik)/M_i, i.e. precision is proportional to M_i.

With one rating per item (M = 1) every weight is 1 and each loss reduces to its
familiar single-label form: `ce` is ordinary cross entropy on the observed
label, `rps` the ranked probability score against a one-hot CDF.
"""

from __future__ import annotations

import torch


def _weights(counts: torch.Tensor, weight_by_M: bool) -> torch.Tensor:
    M = counts.sum(-1)
    if not weight_by_M:
        return torch.ones_like(M)
    return M


def soft_ce_loss(
    log_pmf: torch.Tensor, counts: torch.Tensor, weight_by_M: bool = True
) -> torch.Tensor:
    """Mean M-weighted soft-label cross entropy.

    With weight_by_M=True this is the multinomial NLL per annotation:
    Sum_i Sum_k c_ik log pi_ik  /  Sum_i M_i.
    """
    counts = counts.to(log_pmf.dtype)
    phat = counts / counts.sum(-1, keepdim=True)
    per_item = -(phat * log_pmf).sum(-1)
    w = _weights(counts, weight_by_M)
    return (w * per_item).sum() / w.sum()


def rps_loss(
    log_pmf: torch.Tensor, counts: torch.Tensor, weight_by_M: bool = True
) -> torch.Tensor:
    """Mean M-weighted ranked probability score, Sum_k (F_k - Ghat_k)^2.

    Note we deliberately do NOT subtract the debiasing offset here: it does not
    depend on the prediction, so it contributes nothing to the gradient. It
    matters only for reporting (metrics.rps_debiased).
    """
    counts = counts.to(log_pmf.dtype)
    phat = counts / counts.sum(-1, keepdim=True)
    F_pred = torch.cumsum(log_pmf.exp(), dim=-1)[..., :-1]
    G_emp = torch.cumsum(phat, dim=-1)[..., :-1]
    per_item = ((F_pred - G_emp) ** 2).sum(-1)
    w = _weights(counts, weight_by_M)
    return (w * per_item).sum() / w.sum()


def brier_pmf_loss(
    log_pmf: torch.Tensor, counts: torch.Tensor, weight_by_M: bool = True
) -> torch.Tensor:
    """Mean M-weighted Brier (quadratic) score on the pmf simplex,
    Sum_k (pi_k - phat_k)^2.

    The PMF-domain counterpart of `rps_loss`: RPS is the same quadratic distance
    taken on the CDF, this one takes it on the pmf. Like RPS it is a proper
    scoring rule but not a likelihood.

    As in `rps_loss` we deliberately do NOT subtract the finite-M debiasing
    offset. The offset is Sum_k phat_k(1-phat_k)/(M-1) plus cross terms; it does
    not depend on the prediction, so it drops out of the gradient and matters
    only for reporting.
    """
    counts = counts.to(log_pmf.dtype)
    phat = counts / counts.sum(-1, keepdim=True)
    per_item = ((log_pmf.exp() - phat) ** 2).sum(-1)
    w = _weights(counts, weight_by_M)
    return (w * per_item).sum() / w.sum()


def threshold_bce_loss(
    log_pmf: torch.Tensor, counts: torch.Tensor, weight_by_M: bool = True
) -> torch.Tensor:
    """Mean M-weighted log score on the threshold CDF:

        L_i = - Sum_{b=1}^{K-1} [ Ghat_b log F_b + (1 - Ghat_b) log(1 - F_b) ]

    with F_b = P(Y <= b) the predicted CDF and Ghat_b = cumsum(phat)_b the
    empirical one, exactly as in `rps_loss`.

    Properties, and why they matter here:

    (a) STRICTLY PROPER. Each term is the proper Bernoulli log score for the
        event {Y <= b}; the K-1 thresholds jointly pin the whole distribution, so
        the sum is minimised only at F = Ghat, i.e. pi = phat.

    (b) It equals RPS with a Schervish reweighting. Both losses integrate a
        Bernoulli score over the same K-1 thresholds; RPS uses the Brier
        (quadratic) score, weight 1 at every threshold, and this one uses the log
        score, whose local weight is 1 / (F_b (1 - F_b)). That reweighting - up
        near the tails, flat in the middle - is the intended contrast between the
        two arms.

    (c) It is a COMPOSITE (pseudo-) likelihood, not a likelihood. The K-1
        threshold terms are scored as if independent, but the indicators
        1{Y <= b} are strongly correlated (nested events). So there is no single
        multinomial whose NLL this is, and its Gauss-Newton curvature is NOT
        the Fisher information for this objective. Any Laplace posterior built
        around a tbce-trained head would understate the epistemic term (the
        Godambe "sandwich" correction is missing); use it for MAP fits only.

    Implementation notes:

    * Both log-probabilities are taken in log space with `torch.logcumsumexp`,
      never as `log(cumsum(exp(.)))`. At M = 8 the empirical Ghat_b hits exactly
      0 and 1 all the time, so the loss reduces to a bare log F_b or log(1-F_b)
      there, and the finite-difference of a clamped `log(1-F)` is precisely where
      a cumsum-then-log route goes quietly wrong in the tails.
    * No epsilon clamp on F_b. The logcumsumexp route is exact and numerically
      stable for any finite `log_pmf`, and a floor would bias the tail
      probabilities - the regime this loss exists to probe.
    """
    counts = counts.to(log_pmf.dtype)
    phat = counts / counts.sum(-1, keepdim=True)
    G_emp = torch.cumsum(phat, dim=-1)[..., :-1]                            # Ghat_b
    log_F = torch.logcumsumexp(log_pmf, dim=-1)[..., :-1]                   # log P(Y <= b)
    log_S = torch.logcumsumexp(log_pmf.flip(-1), dim=-1).flip(-1)[..., 1:]  # log P(Y > b)
    per_item = -(G_emp * log_F + (1.0 - G_emp) * log_S).sum(-1)
    w = _weights(counts, weight_by_M)
    return (w * per_item).sum() / w.sum()


LOSSES = {
    "ce": soft_ce_loss,
    "rps": rps_loss,
    "brier": brier_pmf_loss,
    "tbce": threshold_bce_loss,
}


def build_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}, expected one of {sorted(LOSSES)}")
    return LOSSES[name]
