"""Post-hoc recalibration of ordinal PMFs.

The input contract is always "a PMF over K ordinal bins" - a (n, K) array of
probabilities, plus the (n, K) empirical annotator COUNTS it is scored
against. That is deliberate: it is the one interface every method in this
pipeline produces a value for (the softmax/CLM heads' `.exp()` output, the
baselines, the zero-shot `logit_rollout` pmf and the verbal-elicitation
histogram), so one recalibrator serves all of them without refitting anything.

Two families, each with a 1- and 2-parameter variant:

  pmf   smooth  p <- (p + alpha) / (1 + K*alpha)          [alpha fixed, tiny]
        then    p' = softmax(beta * log(p) + gamma * i),  i = 0..K-1
        1-param variant fixes gamma = 0. A scalar ADDITIVE shift on the
        logits would be a no-op (softmax is shift-invariant), which is why
        the 2-param variant multiplies the level index instead - that is a
        real degree of freedom (it tilts the distribution toward high or low
        levels), unlike a constant offset.

  cdf   F = cumsum(p)[:, :K-1], clipped to (eps, 1-eps)
        a = logit(F)
        F' = sigmoid((a - b) / T)
        p' = diff(concat([0], F', [1]))
        1-param variant fixes b = 0. Monotone in F for any T > 0, so F' is
        automatically a valid CDF and p' is automatically a valid pmf -
        no renormalisation needed.

Both families reparameterise their scale as T = exp(theta[0]) (beta for pmf,
T for cdf) so the optimiser sees an unconstrained real vector; theta[1], when
present, is the shift (gamma or b) used additively and needs no
reparameterisation.

NAMING TRAP. `params()` maps theta back to a (T, b) pair using the same names
for both families, but for the `pmf` family its `T` is BETA - the MULTIPLIER on
log p - and the temperature is its reciprocal (beta > 1 sharpens). Use
`describe()`, which names both unambiguously, for anything a person will read.

Plain temperature scaling is family `pmf`, 1 parameter, objective `ce` (the
default of the CLI below). At M = 1 that is exactly the classic single-label
temperature scaling of Guo et al.

Fitting reuses `losses.soft_ce_loss` / `losses.rps_loss` verbatim - the same
M-weighted objectives the heads are trained with.

CLI: fit on the VALIDATION split of every arm in a run's preds files (plus
verbal), apply to val and test, and write the recalibrated arms beside the raw
ones:

    python -m ojc.recalibrate --run-dir runs/<slug>
    python -m ojc.evaluate    --run-dir runs/<slug> --preds-subdir preds_recal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ojc import losses as L

EPS = 1e-12
ALPHA_SMOOTH = 1e-6   # pmf-family additive smoothing before log
CLIP_EPS = 1e-6        # cdf-family clip before logit


# --------------------------------------------------------------------------- #
# the two families
# --------------------------------------------------------------------------- #

def _split_theta(theta: torch.Tensor, n_params: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n_params not in (1, 2):
        raise ValueError(f"n_params must be 1 or 2, got {n_params}")
    scale = theta[0].exp()
    shift = theta[1] if n_params == 2 else theta.new_zeros(())
    return scale, shift


def pmf_transform(p: torch.Tensor, theta: torch.Tensor, n_params: int) -> torch.Tensor:
    """p -> softmax(beta * log(smooth(p)) + gamma * i)."""
    beta, gamma = _split_theta(theta, n_params)
    K = p.shape[-1]
    p_s = (p + ALPHA_SMOOTH) / (1.0 + K * ALPHA_SMOOTH)
    i = torch.arange(K, dtype=p.dtype, device=p.device)
    logits = beta * p_s.log() + gamma * i
    return torch.softmax(logits, dim=-1)


def cdf_transform(p: torch.Tensor, theta: torch.Tensor, n_params: int) -> torch.Tensor:
    """cumsum(p) -> sigmoid((logit(F) - b) / T), re-differenced."""
    T, b = _split_theta(theta, n_params)
    F = torch.cumsum(p, dim=-1)[..., :-1].clamp(CLIP_EPS, 1.0 - CLIP_EPS)
    a = torch.logit(F)
    Fp = torch.sigmoid((a - b) / T)
    ones = torch.ones_like(Fp[..., :1])
    zeros = torch.zeros_like(Fp[..., :1])
    full = torch.cat([zeros, Fp, ones], dim=-1)
    return (full[..., 1:] - full[..., :-1]).clamp_min(0.0)


FAMILIES = {"pmf": pmf_transform, "cdf": cdf_transform}
OBJECTIVES = {"ce": L.soft_ce_loss, "rps": L.rps_loss}


def apply(p, theta, family: str, n_params: int) -> torch.Tensor:
    """Recalibrate a pmf. Returns a torch.float64 tensor of the same shape."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}, got {family!r}")
    p_t = torch.as_tensor(p, dtype=torch.float64)
    th_t = torch.as_tensor(theta, dtype=torch.float64)
    if th_t.numel() != n_params:
        raise ValueError(f"theta has {th_t.numel()} entries, expected {n_params}")
    return FAMILIES[family](p_t, th_t, n_params)


def params(theta, n_params: int) -> dict:
    """Unconstrained theta -> the constrained (T, b) the report table shows.

    Named T/b for BOTH families (beta plays T's role in the pmf family,
    gamma plays b's role) since that is how the calling report tabulates
    them; see the module docstring.
    """
    theta = np.asarray(theta, dtype=np.float64)
    return {"T": float(np.exp(theta[0])),
            "b": float(theta[1]) if n_params == 2 else 0.0}


def describe(theta, family: str, n_params: int) -> dict:
    """The fitted parameters under names that cannot be misread.

    pmf: beta (multiplier on log p), temperature = 1/beta, gamma (level tilt)
    cdf: temperature T on the logit-CDF scale, shift b
    In both, temperature > 1 FLATTENS the prediction and < 1 sharpens it.
    """
    p = params(theta, n_params)
    if family == "pmf":
        return {"beta": p["T"], "temperature": 1.0 / p["T"], "gamma": p["b"]}
    return {"temperature": p["T"], "shift_b": p["b"]}


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #

def fit(p, counts, family: str, n_params: int, objective: str,
        inits: list | None = None, max_iter: int = 200) -> dict:
    """Fit theta minimising the M-weighted `objective` against `counts`.

    theta is 1- or 2-dimensional, so a handful of LBFGS restarts from
    different scale inits (default: identity, sharpen, soften) is cheap
    insurance against a bad basin rather than a real cost.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}, got {objective!r}")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}, got {family!r}")
    p_t = torch.as_tensor(np.asarray(p), dtype=torch.float64)
    c_t = torch.as_tensor(np.asarray(counts), dtype=torch.float64)
    loss_fn = OBJECTIVES[objective]
    transform = FAMILIES[family]

    if inits is None:
        base = [0.0, 0.0] if n_params == 2 else [0.0]
        inits = [base, [0.7] + base[1:], [-0.7] + base[1:]]

    def objective_value(theta: torch.Tensor) -> torch.Tensor:
        pp = transform(p_t, theta, n_params)
        return loss_fn(pp.clamp_min(EPS).log(), c_t)

    best_theta, best_loss = None, float("inf")
    for th0 in inits:
        theta = torch.tensor(np.asarray(th0, dtype=np.float64)[:n_params], requires_grad=True)
        opt = torch.optim.LBFGS([theta], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = objective_value(theta)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            final = float(objective_value(theta))
        if final < best_loss:
            best_theta, best_loss = theta.detach().numpy().copy(), final

    return {"theta": best_theta, "loss": best_loss, "family": family,
            "n_params": n_params, "objective": objective,
            "params": params(best_theta, n_params)}


def cross_fit(p, counts, family: str, n_params: int, objective: str,
             n_folds: int = 5, seed: int = 0) -> dict:
    """M-fold cross-fitting: fit on M-1 folds, predict the held-out fold, pool.

    Returns the pooled recalibrated pmf in the ORIGINAL row order (every
    item is predicted exactly once, by a fit that never saw it), plus the
    per-fold thetas so the spread across folds is visible. Use this for
    evaluation metrics; use `fit` on the full split for the headline T-hat/
    b-hat a report tabulates, per the module docstring's separation of
    concerns.
    """
    p = np.asarray(p, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    n = len(p)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n < n_folds:
        raise ValueError(f"n={n} items is fewer than n_folds={n_folds}")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    fold_id = np.empty(n, dtype=int)
    for f, idx in enumerate(np.array_split(perm, n_folds)):
        fold_id[idx] = f

    pooled = np.empty_like(p)
    thetas, losses = [], []
    for f in range(n_folds):
        held = fold_id == f
        train = ~held
        res = fit(p[train], counts[train], family, n_params, objective)
        thetas.append(res["theta"])
        losses.append(res["loss"])
        pooled[held] = apply(p[held], res["theta"], family, n_params).numpy()

    thetas = np.stack(thetas)
    return {"pooled": pooled, "thetas": thetas,
            "theta_mean": thetas.mean(0), "theta_std": thetas.std(0, ddof=1),
            "fold_losses": losses, "family": family, "n_params": n_params,
            "objective": objective, "n_folds": n_folds, "fold_id": fold_id}


# --------------------------------------------------------------------------- #
# sanity flags
# --------------------------------------------------------------------------- #

def double_softmax_threshold(K: int) -> float:
    """e / (K-1+e): the max class probability a bare unit-logit separation
    (one class at logit 1, the rest at 0) would produce in a K-way softmax.

    A fitted T below this is scaling log-probabilities down to something
    flatter than that minimal separation would allow - implausible as a
    genuine recalibration and much more often the sign that the input pmf
    had already been through an extra, unintended softmax before it reached
    this module.
    """
    return float(np.e / (K - 1 + np.e))


def flag_double_softmax(T_hat: float, K: int) -> bool:
    return float(T_hat) < double_softmax_threshold(K)


# --------------------------------------------------------------------------- #
# CLI: fit on val, apply to val and test, for every arm
# --------------------------------------------------------------------------- #

def recalibrate_blob(blob: dict, family: str, n_params: int, objective: str,
                     arms=None, suffix: str = "+ts") -> tuple[dict, list[dict]]:
    """Return a copy of a preds blob with `<arm><suffix>` added for every pmf
    arm (raw arms kept, so one file scores before and after), plus one record
    of the fitted parameters per arm. Fit on val ONLY; test never informs it."""
    from ojc.evaluate import pmf_arms

    cv = np.asarray(blob["counts_val"], dtype=np.float64)
    out = {**blob, "pred_val": dict(blob["pred_val"]), "pred_test": dict(blob["pred_test"]),
           "recal": {"family": family, "n_params": n_params, "objective": objective,
                     "fit_split": "val", "suffix": suffix}}
    records = []
    K = int(blob["K"])
    for arm in pmf_arms(blob):
        if arms and arm not in arms:
            continue
        pv = np.asarray(blob["pred_val"][arm], dtype=np.float64)
        pt = np.asarray(blob["pred_test"][arm], dtype=np.float64)
        if np.ptp(pv, axis=0).max() < 1e-9:
            # climatology: the same pmf for every item carries no per-item
            # confidence to calibrate, and the fit just drifts toward uniform
            print(f"  [{blob['file']}] {arm:24s} skipped (constant prediction)", flush=True)
            continue
        res = fit(pv, cv, family, n_params, objective)
        out["pred_val"][arm + suffix] = apply(pv, res["theta"], family, n_params).numpy().astype(np.float32)
        out["pred_test"][arm + suffix] = apply(pt, res["theta"], family, n_params).numpy().astype(np.float32)
        d = describe(res["theta"], family, n_params)
        rec = {"file": blob["file"], "task": blob["task"], "arm": arm,
               "family": family, "n_params": n_params, "objective": objective,
               "theta": [float(t) for t in res["theta"]], **d,
               "val_loss_after": float(res["loss"]),
               "n_val": int(len(pv))}
        if family == "pmf":
            rec["double_softmax_flag"] = bool(flag_double_softmax(d["beta"], K))
        records.append(rec)
        print(f"  [{blob['file']}] {arm:24s} " +
              " ".join(f"{k}={v:.3f}" for k, v in d.items()), flush=True)
    return out, records


def main():
    from ojc.evaluate import attach_verbal, load_preds

    ap = argparse.ArgumentParser(description="Fit a recalibrator on val for every arm, "
                                             "apply it to val and test.")
    ap.add_argument("--run-dir", required=True, help="runs/<slug>")
    ap.add_argument("--preds-subdir", default="preds")
    ap.add_argument("--tasks", nargs="*", default=None, help="restrict to these file stems")
    ap.add_argument("--arms", nargs="*", default=None, help="restrict to these arms")
    ap.add_argument("--family", choices=sorted(FAMILIES), default="pmf")
    ap.add_argument("--n-params", type=int, choices=[1, 2], default=1)
    ap.add_argument("--objective", choices=sorted(OBJECTIVES), default="ce")
    ap.add_argument("--no-verbal", action="store_true")
    ap.add_argument("--out-subdir", default=None,
                    help="default <preds-subdir>_recal (preds -> preds_recal)")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    pdir = run_dir / a.preds_subdir
    odir = run_dir / (a.out_subdir or f"{a.preds_subdir}_recal")
    files = sorted(pdir.glob("*.pt"))
    if a.tasks:
        files = [f for f in files if f.stem in a.tasks]
    if not files:
        raise SystemExit(f"no preds files in {pdir}")
    print(f"recalibrating {len(files)} preds files: family={a.family} "
          f"n_params={a.n_params} objective={a.objective} (fit on val)", flush=True)

    odir.mkdir(parents=True, exist_ok=True)
    records = []
    for f in files:
        blob = load_preds(f)
        if not a.no_verbal:
            attach_verbal(blob, run_dir)
        new, recs = recalibrate_blob(blob, a.family, a.n_params, a.objective, arms=a.arms)
        torch.save(new, odir / f.name)
        records += recs
    out = odir / "recal_params.json"
    json.dump(records, open(out, "w"), indent=2)
    print(f"wrote {len(files)} files to {odir} and {out}")


if __name__ == "__main__":
    main()
