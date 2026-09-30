"""Fit the heads and baselines on saved activations, at FULL embedding width.

Every arm reads the same frozen last-layer activation file (written by
`ojc.extract`), the same train/val/test split, and the same M-weighting
convention:

  climatology          training marginal (pooled over annotations), no features
  climatology_itemavg  the same, averaged per item instead
  logit_rollout        the backbone's own answer-position pmf over the digits
                       0..K-1, read out of the activation file - zero-shot,
                       nothing fitted
  spike_0.9            M-weighted ridge -> panel mean -> 0.9 on the nearest level
  spike_tuned          the same, with alpha tuned on validation
  softmax_map__<loss>  linear -> softmax head, trained on the activation
  ordinal_map__<loss>  cumulative-link head, trained on the activation

The two trained heads are each fit under one or more of four objectives, all
M-weighted (see losses.py): soft-label cross entropy (`ce`, the multinomial
likelihood), `rps` (its GLS analogue), the pmf-domain Brier score (`brier`) and
the threshold-BCE composite log score (`tbce`). The default is ce and rps.

No PCA. The heads sit on the raw d-dimensional activation (standardised on
train) with an explicit L2 ridge on the weight matrix whose lambda is selected
on validation loss - see `select_lambda`. Every fit has a fixed optimiser
budget and no early stopping, so "lambda tuned on validation" means what it
says.

Single-label data (M = 1) needs nothing special here: the losses reduce to
their single-label forms and `evaluate.arm_metrics` switches to point metrics.

Outputs, under the run directory (default: the parent of --act-dir):
  results<_tag>.json           one row per (task, arm) with val and test metrics
  preds<_tag>/<task>.pt        per-item val+test pmfs for every arm, with counts
                               and item_id, so any metric can be recomputed or
                               any arm recalibrated later without refitting
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ojc import heads as H
from ojc import losses as L
from ojc import metrics as mt
from ojc import ridge as R
from ojc.evaluate import arm_metrics

# Nine decades of L2 penalty, in MEAN-loss units. The relevant scale is the
# prior precision it implies, n_eff * lam (n_eff = number of training
# annotations), so the grid spans "effectively unregularised" to "an order of
# magnitude tighter than the data" for any corpus from ~1e3 to ~1e5
# annotations. A selection landing on either endpoint is flagged in the output
# rather than silently accepted - that is the signal the grid was too narrow.
LAMBDAS = np.logspace(-5, 3, 9)
PEN_PARAM = {"softmax": "linear.weight", "clm": "w.weight"}
HEAD_LABEL = {"softmax": "softmax", "clm": "ordinal"}

# L2 penalty, in SUM units, on the parameters the ridge does NOT penalise
# (biases, cutpoints). A gauge fix: adding the same constant to every softmax
# bias leaves the pmf unchanged, so that direction is otherwise unidentified.
# It costs PRIOR_FLOOR/n_eff of the mean objective - negligible - and it is
# part of the MAP objective, so removing it changes the fitted heads.
PRIOR_FLOOR = 1.0


def _pin_threads() -> int:
    """The fits are not thread-invariant in their last bits (reduction order),
    and a fixed Adam budget can amplify that into the third decimal of a
    reported loss. Pin the thread count so a rerun reproduces exactly;
    OJC_THREADS=0 opts out. More threads are also SLOWER at these shapes."""
    n = int(os.environ.get("OJC_THREADS", "8"))
    if n <= 0:
        return torch.get_num_threads()
    n = min(n, os.cpu_count() or n)
    torch.set_num_threads(n)
    return n


THREADS = _pin_threads()


def _device(arg: str | None) -> str:
    if arg:
        return arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# MAP fit with an explicit L2 ridge
# --------------------------------------------------------------------------- #

def _penalty(head, pen_name: str, lam: float, n_eff: float) -> torch.Tensor:
    """(lam/2)||W||^2 on the ridge weights + the gauge-fixing floor on the rest,
    both in MEAN units so they can be added straight to loss_fn's output."""
    pen = 0.0
    for n, p in head.named_parameters():
        pen = pen + (lam if n == pen_name else PRIOR_FLOOR / n_eff) * (p ** 2).sum()
    return 0.5 * pen


def fit_l2(head, Xtr, ctr, loss_fn, lam: float, pen_name: str,
           steps: int = 2000, lr: float = 1e-2, log_every: int = 200) -> dict:
    """Minimise  mean_loss(theta) + (lam/2)||W||^2  for a FIXED budget.

    No early stopping, on purpose. Stopping on validation loss would be a second,
    implicit regulariser layered on top of the explicit one. Every lambda on the
    path gets exactly the same optimiser budget, so the comparison between them
    is a comparison of penalties and nothing else.

    The penalty covers the weight matrix only. Biases and cutpoints are location
    parameters; shrinking them toward zero would say the model prefers level 0.

    `grad_norm` and `rel_change_tail` are returned so a reader can check the
    budget was actually enough rather than take it on trust.
    """
    n_eff = float(ctr.sum())
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-2)
    trace = []
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        base = loss_fn(head(Xtr), ctr, True)
        obj = base + _penalty(head, pen_name, lam, n_eff)
        obj.backward()
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in head.parameters() if p.grad is not None))
        opt.step()
        sched.step()
        if t % log_every == 0 or t == steps - 1:
            trace.append([t, float(base.detach()), float(obj.detach()), float(gn.detach())])
    with torch.no_grad():
        base = float(loss_fn(head(Xtr), ctr, True))
        pen = float(_penalty(head, pen_name, lam, n_eff))
    rel = abs(trace[-1][2] - trace[-2][2]) / max(abs(trace[-1][2]), 1e-12) if len(trace) > 1 else np.nan
    return {"train_loss": base, "train_obj": base + pen, "penalty": pen,
            "grad_norm": float(trace[-1][3]), "rel_change_tail": float(rel),
            "steps": steps, "lr": lr, "lam": float(lam), "trace": trace}


def build(head_name: str, d: int, K: int, seed: int, dev: str):
    torch.manual_seed(seed)
    return H.build_head(head_name, d, K).double().to(dev)


def select_lambda(head_name, d, K, Xtr, ctr, Xva, cva, loss_fn, lambdas,
                  steps, select_steps, lr, seed, dev, finalists=2, verbose=True) -> dict:
    """Choose lambda on validation loss, then refit the finalists at full budget.

    Ranking a grid that spans eight decades is coarse work, so the path is fit
    at a reduced budget (`select_steps`). Only the top `finalists` are refit at
    the full budget and re-compared, which is what actually decides the winner;
    the reported `margin_over_runner_up` says how close the call was.

    Deliberately NOT warm-started down the path: every fit starts from the same
    seeded initialisation, so the selected model is reproducible from its own
    lambda alone.

    Returns the winning fitted head, so nothing is fit a third time.
    """
    rows = []
    for lam in lambdas:
        head = build(head_name, d, K, seed, dev)
        info = fit_l2(head, Xtr, ctr, loss_fn, float(lam), PEN_PARAM[head_name],
                      steps=select_steps, lr=lr)
        with torch.no_grad():
            info["val_loss"] = float(loss_fn(head(Xva), cva, True))
        rows.append(info)
        if verbose:
            print(f"      lam={lam:9.1e}  train={info['train_loss']:.5f} "
                  f"val={info['val_loss']:.5f} |g|={info['grad_norm']:.2e}", flush=True)
        del head

    order = list(np.argsort([r["val_loss"] for r in rows]))
    finals = []
    for j in order[:max(1, finalists)]:
        head = build(head_name, d, K, seed, dev)
        info = fit_l2(head, Xtr, ctr, loss_fn, float(lambdas[j]), PEN_PARAM[head_name],
                      steps=steps, lr=lr)
        with torch.no_grad():
            info["val_loss"] = float(loss_fn(head(Xva), cva, True))
        finals.append((info["val_loss"], j, head, info))
        if verbose:
            print(f"      refit lam={lambdas[j]:9.1e} at {steps} steps -> "
                  f"val={info['val_loss']:.6f}", flush=True)
    finals.sort(key=lambda t: t[0])
    best_val, j, head, info = finals[0]
    margin = (finals[1][0] - best_val) if len(finals) > 1 else float("nan")
    return {"head": head, "fit": info, "lam": float(lambdas[j]), "best_index": int(j),
            "at_grid_edge": int(j) in (0, len(lambdas) - 1),
            "margin_over_runner_up": float(margin),
            "select_steps": select_steps, "finalists": len(finals),
            "finalist_lams": [float(lambdas[t[1]]) for t in finals],
            "path": [{k: v for k, v in r.items() if k != "trace"} for r in rows]}


# --------------------------------------------------------------------------- #
# split helpers
# --------------------------------------------------------------------------- #

def align_rollout(logit_pmf: np.ndarray, reverse: bool) -> tuple[np.ndarray, bool]:
    """Put the model's answer on the same level direction as the stored counts.

    Needed only when a prompt's scale points the opposite way to the data (the
    dataset spec's `reverse_rollout`; in the paper corpora only `mhs:respect`,
    whose prompt asks whether a comment IS respectful while its counts are a
    disrespect scale). Every trained arm absorbs such a convention while
    fitting - RPS and MAE are invariant when prediction and target reverse
    together - but the zero-shot rollout has no fitting step in which to
    discover it. The model's pmf is reversed, never the counts.
    """
    if not reverse:
        return logit_pmf, False
    return np.ascontiguousarray(logit_pmf[:, ::-1]), True


def subsample_train(tr: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """A NESTED down-sample of the training mask, for training-size sweeps.

    ONE permutation of the training indices, seeded by `seed` alone and never by
    `frac`, and every fraction takes a PREFIX of it - so for a fixed seed the
    25% subset is a subset of the 50%, and a difference between two fractions is
    about how much data was added, not which items were drawn. Validation and
    test stay at full size. `frac >= 1.0` returns the mask untouched.
    """
    if frac >= 1.0:
        return tr
    if not 0.0 < frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1]; got {frac}")
    idx = np.flatnonzero(tr)
    k = int(round(frac * len(idx)))
    if k < 1:
        raise ValueError(f"train_frac={frac} leaves {k} training items")
    keep = np.sort(idx[np.random.RandomState(seed).permutation(len(idx))[:k]])
    out = np.zeros_like(tr)
    out[keep] = True
    return out


def holdout_split(tr: np.ndarray, va: np.ndarray, te: np.ndarray,
                  group_id: np.ndarray, groups) -> tuple:
    """Domain-holdout split, for covariate-shift experiments.

    Every item whose `group_id` is in `groups` becomes TEST whatever it was
    before, and is removed from train AND val (leaving it in val would let
    lambda be selected against the domain the fit is supposed never to have
    seen). The original in-domain test items stay, so one fit yields both
    blocks; `holdout_test` in the preds file separates them.
    """
    want = set(groups)
    if not want:
        raise ValueError("holdout group list is empty")
    is_d = np.array([g in want for g in group_id], dtype=bool)
    if not is_d.any():
        raise ValueError(f"none of the {len(want)} holdout groups match this task")
    if (tr & ~is_d).sum() < 1 or (va & ~is_d).sum() < 1:
        raise ValueError("holding out this domain leaves an empty train or val split")
    return tr & ~is_d, va & ~is_d, te | is_d, is_d


# --------------------------------------------------------------------------- #
# one task
# --------------------------------------------------------------------------- #

def run_task(path: Path, dev: str, lambdas, steps, select_steps, lr, seed,
             pred_dir: Path, train_frac: float = 1.0, train_subsample_seed: int = 0,
             fit_split: str = "train", fit_n: int | None = None,
             heads=("softmax", "clm"), losses=("ce", "rps"),
             holdout_groups=None, n_boot_null: int = 200) -> list[dict]:
    blob = torch.load(path, weights_only=False)
    X = blob["X"].double()
    counts = blob["counts"].double()
    split = np.array(blob["split"])
    K = int(blob["K"])
    task = blob["task"]
    # outputs are named after the FILE (colon-free), not the task name
    stem = path.stem
    backbone = blob["model"]
    item_id = np.array(blob["item_id"])
    logit_pmf = blob["logit_pmf"].double().numpy()

    tr, va, te = split == "train", split == "val", split == "test"
    is_holdout = None
    if holdout_groups:
        # before subsample_train, so --train-frac keeps meaning "a fraction of
        # what THIS arm trains on" rather than of the full corpus
        tr, va, te, is_holdout = holdout_split(tr, va, te,
                                               np.array(blob["group_id"]), holdout_groups)
        print(f"  holdout: {int(is_holdout.sum())} items -> test "
              f"({int(tr.sum())} train / {int(va.sum())} val left)", flush=True)
    if fit_split == "val":
        # Fit on the VAL split; TRAIN takes over val's role (lambda selection,
        # ridge-val fit, spike alpha tuning, reported `val` metrics). TEST never
        # moves. Pass an explicit --lambdas: there is nothing held out to select on.
        tr, va = va, tr
    n_train_full = int(tr.sum())
    if fit_n is not None:
        if not 1 <= fit_n <= n_train_full:
            raise ValueError(f"--fit-n={fit_n} is outside 1..{n_train_full} "
                             f"(the {fit_split} split)")
        frac = 1.0 if fit_n == n_train_full else fit_n / n_train_full
        tr = subsample_train(tr, frac, train_subsample_seed)
        if int(tr.sum()) != fit_n:                       # rounding sanity check
            raise ValueError(f"--fit-n={fit_n} rounded to {int(tr.sum())} items")
    else:
        tr = subsample_train(tr, train_frac, train_subsample_seed)
    if not (tr.any() and va.any() and te.any()):
        raise ValueError(f"{path}: empty split (train/val/test = "
                         f"{int(tr.sum())}/{int(va.sum())}/{int(te.sum())})")
    # the standardiser is fit on the (sub)sample actually trained on
    proj, _ = R.standardizer(X[tr])
    Xtr, Xva, Xte = (proj(X[m]).to(dev) for m in (tr, va, te))
    ctr, cva, cte = (counts[m].to(dev) for m in (tr, va, te))
    ctr_np, cva_np, cte_np = (counts[m].numpy() for m in (tr, va, te))
    d = X.shape[1]
    n_eff = float(ctr.sum())
    M = counts.sum(-1).numpy()
    print(f"  {task}: K={K} d={d} n=({int(tr.sum())},{int(va.sum())},{int(te.sum())}) "
          f"n_eff_train={n_eff:.0f} M in [{M.min():.0f}, {M.max():.0f}] "
          f"-> {mt.regime(counts.numpy())} metrics", flush=True)

    preds = {"val": {}, "test": {}}
    rows = []

    def record(arm: str, Pva: np.ndarray, Pte: np.ndarray, extra: dict):
        preds["val"][arm] = np.asarray(Pva, dtype=np.float32)
        preds["test"][arm] = np.asarray(Pte, dtype=np.float32)
        rows.append({"backbone": backbone, "task": task, "file": stem,
                     "K": K, "d": d, "arm": arm,
                     "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
                     "n_eff_train": n_eff,
                     "n_train_full": n_train_full,
                     "train_frac": float(train_frac),
                     "train_subsample_seed": int(train_subsample_seed),
                     "fit_split": fit_split,
                     "fit_n": (int(fit_n) if fit_n is not None else None),
                     **extra,
                     "test": arm_metrics(Pte, cte_np, n_boot_null=n_boot_null),
                     "val": arm_metrics(Pva, cva_np, n_boot_null=n_boot_null)})

    def headline(r: dict) -> str:
        t = r["test"]
        if t["rps_debiased"] is not None:
            return f"rps_debiased={t['rps_debiased']:+.4f}"
        return f"rps={t['rps']:.4f} acc={t['acc']:.3f}"

    # ---- climatology ------------------------------------------------------ #
    for name, P in R.climatology_both(ctr_np, int(te.sum())).items():
        Pva = R.climatology(ctr_np, int(va.sum()), pooled=(name == "climatology"))
        record(name, Pva, P, {"family": "baseline", "head": None, "loss": None})

    # ---- logit rollout ---------------------------------------------------- #
    logit_pmf, flipped = align_rollout(logit_pmf, bool(blob.get("reverse_rollout", False)))
    record("logit_rollout", logit_pmf[va], logit_pmf[te],
           {"family": "baseline", "head": None, "loss": None,
            "logit_pmf_reversed": flipped,
            "note": "backbone's own answer-position pmf; nothing fitted"})

    # ---- ridge -> spike --------------------------------------------------- #
    lv = np.arange(K, dtype=np.float64)
    ytr = torch.tensor((mt.to_pmf(ctr_np) * lv).sum(1), device=dev)
    yva = torch.tensor((mt.to_pmf(cva_np) * lv).sum(1), device=dev)
    rr = R.fit_mean_ridge(Xtr, ytr, ctr.sum(-1), Xva, yva, cva.sum(-1))
    mva = (Xva.double() @ rr["B"] + rr["b"]).cpu().numpy()
    mte = (Xte.double() @ rr["B"] + rr["b"]).cpu().numpy()
    a_star, a_scores = R.tune_alpha(mva, cva_np, K)
    rmeta = {"family": "baseline", "head": "ridge", "loss": None,
             "ridge_lam": rr["lam"], "ridge_lam_at_edge": rr["at_grid_edge"],
             "ridge_val_wmse": rr["val_wmse"], "ridge_val_mae": rr["val_mae"]}
    record("spike_0.9", R.spike(mva, K, 0.9), R.spike(mte, K, 0.9),
           {**rmeta, "alpha": 0.9})
    record("spike_tuned", R.spike(mva, K, a_star), R.spike(mte, K, a_star),
           {**rmeta, "alpha": a_star, "alpha_grid_min": float(a_scores.min())})
    # the ridge mean itself is worth keeping even though it is not a pmf
    preds["val"]["ridge_mean"] = mva.astype(np.float32)
    preds["test"]["ridge_mean"] = mte.astype(np.float32)

    # ---- trained heads ---------------------------------------------------- #
    for head_name in heads:
        for loss_name in losses:
            t0 = time.time()
            label = HEAD_LABEL[head_name]
            print(f"    [{task}] {label}/{loss_name}", flush=True)
            loss_fn = L.build_loss(loss_name)
            sel = select_lambda(head_name, d, K, Xtr, ctr, Xva, cva, loss_fn,
                                lambdas, steps, select_steps, lr, seed, dev)
            lam, head, fit = sel["lam"], sel["head"], sel["fit"]
            with torch.no_grad():
                Pva_map = head(Xva).exp().cpu().numpy()
                Pte_map = head(Xte).exp().cpu().numpy()
            record(f"{label}_map__{loss_name}", Pva_map, Pte_map,
                   {"family": "head", "head": label, "loss": loss_name,
                    "lam": lam, "lam_at_edge": sel["at_grid_edge"],
                    "lam_margin": sel["margin_over_runner_up"],
                    "lam_finalists": sel["finalist_lams"],
                    "lam_path_val": [r["val_loss"] for r in sel["path"]],
                    "lambdas": [float(x) for x in lambdas],
                    "n_params": int(sum(p.numel() for p in head.parameters())),
                    "fit": {k: v for k, v in fit.items() if k != "trace"}})
            print(f"      lam={lam:.1e} {headline(rows[-1])} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    pred_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": backbone, "task": task, "file": stem, "K": K,
                "fit_split": fit_split,
                "fit_n": (int(fit_n) if fit_n is not None else None),
                "reverse_rollout": flipped,
                "item_id_val": item_id[va], "item_id_test": item_id[te],
                "holdout_test": (is_holdout[te] if is_holdout is not None else None),
                "counts_val": cva_np.astype(np.float32),
                "counts_test": cte_np.astype(np.float32),
                "pred_val": preds["val"], "pred_test": preds["test"]},
               pred_dir / f"{stem}.pt")
    return rows


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--act-dir", required=True,
                    help="directory of activation files, e.g. runs/<slug>/activations")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="restrict to these activation file stems")
    ap.add_argument("--run-dir", default=None,
                    help="where results.json and preds/ go (default: parent of --act-dir)")
    ap.add_argument("--device", default=None, help="default cuda:0 if available, else cpu")
    ap.add_argument("--steps", type=int, default=6000,
                    help="optimiser budget for the fits that are actually reported")
    ap.add_argument("--select-steps", type=int, default=1500,
                    help="reduced budget for ranking the lambda path; the finalists "
                         "are refit at --steps before the winner is chosen")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambdas", nargs="*", type=float, default=None,
                    help=f"L2 grid (mean-loss units); default {LAMBDAS.tolist()}")
    ap.add_argument("--heads", nargs="*", default=["softmax", "clm"],
                    choices=["softmax", "clm"])
    ap.add_argument("--losses", nargs="*", default=["ce", "rps"],
                    choices=sorted(L.LOSSES))
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="fit on this (nested) fraction of the TRAIN split only; "
                         "val and test stay at full size")
    ap.add_argument("--train-subsample-seed", type=int, default=0)
    ap.add_argument("--fit-split", choices=["train", "val"], default="train",
                    help="'val' swaps train<->val (needs an explicit --lambdas)")
    ap.add_argument("--fit-n", type=int, default=None,
                    help="fit on exactly this many items (overrides --train-frac)")
    ap.add_argument("--holdout-groups", default=None,
                    help="JSON file with a list of group ids to move to TEST "
                         "(removed from train and val). Requires --tag.")
    ap.add_argument("--n-boot-null", type=int, default=200,
                    help="bootstrap replicates for the ece_thr null floor (0 = skip)")
    ap.add_argument("--tag", default="",
                    help="suffix for results/preds so a variant run sits beside the main one")
    ap.add_argument("--skip-existing", action="store_true",
                    help="resume from results<_tag>.partial.json")
    a = ap.parse_args()

    holdout = None
    if a.holdout_groups:
        if not a.tag:
            ap.error("--holdout-groups requires --tag")
        holdout = json.load(open(a.holdout_groups))
        if not isinstance(holdout, list):
            ap.error(f"{a.holdout_groups} must hold a JSON list of group ids")
    if a.fit_split != "train" and not a.lambdas:
        ap.error("--fit-split val has no held-out set to select lambda on; "
                 "pass an explicit --lambdas")
    if a.fit_split != "train" and a.holdout_groups:
        ap.error("--fit-split val and --holdout-groups cannot be combined")

    fdir = Path(a.act_dir)
    run_dir = Path(a.run_dir) if a.run_dir else fdir.parent
    dev = _device(a.device)
    lambdas = np.array(a.lambdas) if a.lambdas else LAMBDAS
    files = sorted(fdir.glob("*.pt"))
    if a.tasks:
        files = [f for f in files if f.stem in a.tasks]
    if not files:
        raise SystemExit(f"no activation files in {fdir}")

    tag = f"_{a.tag}" if a.tag else ""
    out = run_dir / f"results{tag}.json"
    pred_dir = run_dir / f"preds{tag}"
    part = out.with_suffix(".partial.json")

    rows = []
    if a.skip_existing and part.exists():
        rows = json.load(open(part))
        done = {r["file"] for r in rows}
        files = [f for f in files if f.stem not in done]
        print(f"resuming: {len(done)} tasks already done", flush=True)

    print(f"device={dev} threads={THREADS} steps={a.steps} (select {a.select_steps}) "
          f"lambdas={lambdas.tolist()}", flush=True)
    print(f"preds -> {pred_dir}\n{len(files)} activation files from {fdir}", flush=True)

    for f in files:
        t0 = time.time()
        rows += run_task(f, dev, lambdas, a.steps, a.select_steps, a.lr, a.seed,
                         pred_dir, train_frac=a.train_frac,
                         train_subsample_seed=a.train_subsample_seed,
                         fit_split=a.fit_split, fit_n=a.fit_n,
                         heads=tuple(a.heads), losses=tuple(a.losses),
                         holdout_groups=holdout, n_boot_null=a.n_boot_null)
        run_dir.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(part, "w"), indent=2)
        print(f"  {f.stem} done in {time.time() - t0:.0f}s", flush=True)

    json.dump(rows, open(out, "w"), indent=2)
    if part.exists():
        part.unlink()
    print(f"\nwrote {out}  ({len(rows)} rows)")

    edge = [r for r in rows if r.get("lam_at_edge") or r.get("ridge_lam_at_edge")]
    if edge:
        print(f"WARNING: {len(edge)} rows selected lambda at a grid endpoint "
              f"(widen --lambdas):")
        for r in edge[:20]:
            print(f"  {r['task']:18s} {r['arm']:24s} lam={r.get('lam', r.get('ridge_lam'))}")


if __name__ == "__main__":
    main()
