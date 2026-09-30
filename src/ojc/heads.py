"""Prediction heads over frozen LLM activations.

Two arms:

  SoftmaxHead   unconstrained categorical - can represent ANY pmf on the
                simplex, including bimodal annotator disagreement.
  CLMHead       cumulative link ("ordinal"), one latent score + shared
                cutpoints. Strictly respects the ordering, but its reachable set
                is a ONE-DIMENSIONAL CURVE in the simplex, so it structurally
                cannot represent bimodal disagreement. This is the point of the
                comparison, not a bug.

Both return a LOG-pmf of shape (n, K).

Cutpoint parameterisation:
    alpha_1 = a0,  alpha_k = a0 + sum_{j<k} softplus(delta_j)
The CLM head carries (a0, delta) as free parameters, never alpha itself, so any
parameter vector - including one perturbed by an optimiser or a posterior draw -
gives strictly increasing cutpoints and hence a valid pmf.

Note: `nn.Module.named_parameters` yields a module's own parameters before its
children's, so `CLMHead` flattens as `[a0, delta, w.weight]` - cutpoints first,
not in constructor order. Read the order back off `named_parameters` rather than
assuming it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLAMP = 30.0  # keeps sigmoid out of exact 0/1 in fp32


class SoftmaxHead(nn.Module):
    """Plain linear -> softmax. Ignores the ordering of the levels."""

    def __init__(self, d: int, K: int):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(d, K)
        nn.init.zeros_(self.linear.bias)
        nn.init.normal_(self.linear.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.linear(x), dim=-1)


def _cutpoints(a0: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """(a0, delta) -> strictly increasing alpha of length K-1."""
    steps = F.softplus(delta)
    return torch.cat([a0, a0 + torch.cumsum(steps, dim=0)], dim=0)


def _clm_log_pmf(latent: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """P(y=k) = sigma(alpha_k - f) - sigma(alpha_{k-1} - f), with the ends closed.

    Computed as differences of cumulative probabilities. We clamp the logits and
    floor the differences before the log: adjacent cutpoints can be very close,
    and the subtraction is the one numerically delicate step in the head.
    """
    z = (alpha[None, :] - latent[:, None]).clamp(-CLAMP, CLAMP)
    cum = torch.sigmoid(z)  # (n, K-1)
    ones = torch.ones_like(cum[:, :1])
    zeros = torch.zeros_like(cum[:, :1])
    cum_full = torch.cat([zeros, cum, ones], dim=1)  # (n, K+1)
    p = cum_full[:, 1:] - cum_full[:, :-1]
    return torch.log(p.clamp_min(1e-9))


class CLMHead(nn.Module):
    """Cumulative link (ordinal) head with a single latent score."""

    def __init__(self, d: int, K: int):
        super().__init__()
        self.K = K
        self.w = nn.Linear(d, 1, bias=False)
        nn.init.normal_(self.w.weight, std=0.01)
        # init cutpoints to evenly spaced quantiles of a standard logistic
        self.a0 = nn.Parameter(torch.tensor([-1.0 * (K - 2) / 2.0]))
        self.delta = nn.Parameter(torch.full((K - 2,), 0.5413))  # softplus -> ~1.0

    @property
    def alpha(self) -> torch.Tensor:
        return _cutpoints(self.a0, self.delta)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _clm_log_pmf(self.latent(x), self.alpha)


HEADS = {
    "softmax": SoftmaxHead,
    "clm": CLMHead,
}


def build_head(name: str, d: int, K: int) -> nn.Module:
    if name not in HEADS:
        raise ValueError(f"unknown head {name!r}, expected one of {sorted(HEADS)}")
    return HEADS[name](d, K)
