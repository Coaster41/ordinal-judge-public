"""Checks for the head-training path (`ojc.train_heads`), the ridge baselines,
arm scoring (`ojc.evaluate`) and the recalibration CLI's plumbing.

The end-to-end sections (4d, 4g, 4h, 7) run `run_task` on a toy activation file
and `json.dumps` what it returns: a results file is written in one dump at the
end of a task, so an un-encodable value (np.bool_ is not a bool) would discard
the whole task's work, and an in-memory check cannot catch that.

Section numbers follow the original research suite; gaps covered the Laplace
posterior, which is not shipped here.

Run: python tests/test_heads.py      (or: pytest tests/)
"""

from __future__ import annotations

import json as _json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from ojc import evaluate as EV
from ojc import losses as L
from ojc import metrics as mt
from ojc import recalibrate as RC
from ojc import ridge as R
from ojc import train_heads as S2

torch.set_default_dtype(torch.float64)
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


def toy(K=4, d=6, n=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    rng = np.random.RandomState(seed)
    M = rng.randint(2, 12, size=n)
    counts = torch.tensor(
        np.stack([rng.multinomial(int(M[i]), np.ones(K) / K) for i in range(n)]),
        dtype=torch.float64)
    return X, counts


# --------------------------------------------------------------------------- #
print("\n4. The L2 penalty")
X, counts = toy(K=4, d=6, n=40, seed=3)
norms = []
for lam in (1e-4, 1e-1, 1e2):
    head = S2.build("softmax", 6, 4, 0, "cpu")
    S2.fit_l2(head, X, counts, L.build_loss("ce"), lam, "linear.weight", steps=1500, lr=3e-2)
    norms.append(float(head.linear.weight.norm()))
check("a bigger lambda really does shrink the weights",
      norms[0] > norms[1] > norms[2], f"||W||={[round(v, 3) for v in norms]}")

head = S2.build("clm", 6, 4, 0, "cpu")
S2.fit_l2(head, X, counts, L.build_loss("ce"), 1e3, "w.weight", steps=1500, lr=3e-2)
check("the penalty leaves the cutpoints alone",
      float(head.delta.abs().max()) > 1e-3 and float(head.w.weight.norm()) < 1e-2,
      f"||w||={float(head.w.weight.norm()):.2e} |delta|max={float(head.delta.abs().max()):.3f}")

# --------------------------------------------------------------------------- #
print("\n4b. Lambda selection")
Xs, cs = toy(K=4, d=6, n=60, seed=11)
Xv, cvv = toy(K=4, d=6, n=40, seed=12)
grid = np.logspace(-4, 2, 4)
sel = S2.select_lambda("softmax", 6, 4, Xs, cs, Xv, cvv, L.build_loss("ce"),
                       grid, steps=800, select_steps=200, lr=3e-2, seed=0, dev="cpu",
                       verbose=False)
check("select_lambda returns a lambda from the grid it was given",
      sel["lam"] in [float(g) for g in grid], f"lam={sel['lam']:.3g}")
check("...ranks the whole path and refits exactly the finalists",
      len(sel["path"]) == len(grid) and sel["finalists"] == 2
      and len(sel["finalist_lams"]) == 2)
check("...and flags a selection sitting at a grid endpoint",
      sel["at_grid_edge"] == (sel["best_index"] in (0, len(grid) - 1)))

fresh = S2.build("softmax", 6, 4, 0, "cpu")
finfo = S2.fit_l2(fresh, Xs, cs, L.build_loss("ce"), sel["lam"], "linear.weight",
                  steps=800, lr=3e-2)
check("the head it returns IS the full-budget fit at the lambda it reports",
      float((fresh.linear.weight - sel["head"].linear.weight).abs().max()) == 0.0
      and abs(finfo["train_obj"] - sel["fit"]["train_obj"]) < 1e-15,
      "bit-identical to an independent refit")

sel1 = S2.select_lambda("softmax", 6, 4, Xs, cs, Xv, cvv, L.build_loss("ce"),
                        grid, steps=800, select_steps=200, lr=3e-2, seed=0, dev="cpu",
                        finalists=1, verbose=False)
check("finalists=1 still returns a coherent result",
      sel1["finalists"] == 1 and np.isnan(sel1["margin_over_runner_up"]))

sel2 = S2.select_lambda("softmax", 6, 4, Xs, cs, Xv, cvv, L.build_loss("ce"),
                        grid, steps=800, select_steps=200, lr=3e-2, seed=0, dev="cpu",
                        verbose=False)
check("selection is deterministic across identical calls",
      sel2["lam"] == sel["lam"]
      and float((sel2["head"].linear.weight - sel["head"].linear.weight).abs().max()) == 0.0)

check("the reported margin is the val-loss gap to the runner-up, and is >= 0",
      sel["margin_over_runner_up"] >= 0.0,
      f"margin={sel['margin_over_runner_up']:.3e}")

# A grid whose right answer is unambiguous: pure noise targets, so the heaviest
# penalty must win on validation.
gnz = torch.Generator().manual_seed(21)
Xn = torch.randn(40, 30, generator=gnz, dtype=torch.float64)
cn = toy(K=4, d=6, n=40, seed=13)[1]
Xn2 = torch.randn(40, 30, generator=gnz, dtype=torch.float64)
cn2 = toy(K=4, d=6, n=40, seed=14)[1]
seln = S2.select_lambda("softmax", 30, 4, Xn, cn, Xn2, cn2, L.build_loss("ce"),
                        np.array([1e-6, 1e2]), steps=600, select_steps=200, lr=3e-2,
                        seed=0, dev="cpu", verbose=False)
check("on unlearnable noise the heavy penalty wins on validation",
      seln["lam"] == 100.0, f"lam={seln['lam']:.3g}")

# --------------------------------------------------------------------------- #
print("\n5. Ridge")
for n, d in [(60, 8), (8, 60)]:      # the n>=d and n<d branches are different code
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    w = torch.rand(n, generator=g, dtype=torch.float64) * 5 + 1
    lams = np.array([1e-3, 1.0, 37.0, 1e4])
    B, b = R.ridge_path(X, y, w, lams)
    sw = w.sum()
    xbar = (w[:, None] * X).sum(0) / sw
    ybar = (w * y).sum() / sw
    Xc, yc = X - xbar, y - ybar
    worst_B = worst_b = 0.0
    for j, lam in enumerate(lams):
        A = Xc.T @ (w[:, None] * Xc) + lam * torch.eye(d, dtype=torch.float64)
        beta = torch.linalg.solve(A, Xc.T @ (w * yc))
        worst_B = max(worst_B, float((B[j] - beta).abs().max()))
        worst_b = max(worst_b, abs(float(b[j]) - float(ybar - xbar @ beta)))
    check(f"ridge path == weighted normal equations (n={n}, d={d})",
          worst_B < 1e-7 and worst_b < 1e-7, f"max|dbeta|={worst_B:.1e}")

g = torch.Generator().manual_seed(1)
X = torch.randn(40, 5, generator=g, dtype=torch.float64)
y = torch.randn(40, generator=g, dtype=torch.float64)
w = torch.cat([torch.ones(20, dtype=torch.float64), torch.full((20,), 50.0, dtype=torch.float64)])
a_, _ = R.ridge_path(X, y, torch.ones(40, dtype=torch.float64), np.array([1.0]))
b_, _ = R.ridge_path(X, y, w, np.array([1.0]))
check("the M weights actually change the ridge fit",
      float((a_ - b_).abs().max()) > 1e-3, f"max|d|={float((a_ - b_).abs().max()):.3f}")

g = torch.Generator().manual_seed(2)
X = torch.randn(300, 4, generator=g, dtype=torch.float64)
beta = torch.tensor([1.0, -2.0, 0.5, 0.0], dtype=torch.float64)
B, b = R.ridge_path(X, X @ beta + 3.0, torch.ones(300, dtype=torch.float64), np.array([1e-8]))
check("ridge recovers a noiseless linear signal and its intercept",
      float((B[0] - beta).abs().max()) < 1e-5 and abs(float(b[0]) - 3.0) < 1e-5)

Pk = R.spike(np.array([0.2, 2.9, 4.4]), 5, 0.7)
check("spike is a pmf, peaks at the rounded mean, and holds alpha there",
      np.allclose(Pk.sum(1), 1.0) and Pk.argmax(1).tolist() == [0, 3, 4]
      and np.allclose(Pk.max(1), 0.7))
Pk = R.spike(np.array([1.0]), 5, 0.05)
check("spike below uniform is still a valid pmf",
      np.allclose(Pk.sum(1), 1.0) and (Pk >= 0).all())

counts = np.array([[10.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
check("climatology(pooled) weights ANNOTATIONS, item-average weights ITEMS",
      np.allclose(R.climatology(counts, 1, True)[0], [10 / 12, 2 / 12])
      and np.allclose(R.climatology(counts, 1, False)[0], [1 / 3, 2 / 3]))

g = torch.Generator().manual_seed(9)
Xr = torch.randn(50, 4, generator=g, dtype=torch.float64)
Br, br = R.ridge_path(Xr, torch.randn(50, generator=g, dtype=torch.float64),
                      torch.ones(50, dtype=torch.float64), np.array([0.1, 1.0, 10.0]))
Pr = R.predict(Br, br, Xr)
check("ridge.predict is (n_lambda, n) and agrees row by row with B @ x + b",
      Pr.shape == (3, 50)
      and float((Pr[1] - (Xr @ Br[1] + br[1])).abs().max()) < 1e-12,
      f"shape={tuple(Pr.shape)}")

rsel = R.fit_mean_ridge(Xr, Xr[:, 0] * 2.0, torch.ones(50, dtype=torch.float64),
                        Xr, Xr[:, 0] * 2.0, torch.ones(50, dtype=torch.float64))
check("fit_mean_ridge selects a lambda and reports its own validation criterion",
      rsel["lam"] in rsel["lambdas"] and rsel["val_wmse"] >= 0
      and len(rsel["path_val_wmse"]) == len(rsel["lambdas"])
      and rsel["val_wmse"] == min(rsel["path_val_wmse"]),
      f"lam={rsel['lam']:.3g} wmse={rsel['val_wmse']:.2e}")

counts = np.zeros((200, 5))
counts[:, 2] = 10.0
a_star, scores = R.tune_alpha(np.full(200, 2.0), counts, 5)
check("alpha* goes to the top of the grid on unanimous panels",
      a_star > 0.95 and int(scores.argmin()) == len(scores) - 1, f"alpha*={a_star:.2f}")

counts = np.full((200, 5), 2.0)
a_star, _ = R.tune_alpha(np.full(200, 2.0), counts, 5)
check("...and down toward uniform on maximally split panels",
      a_star < 0.35, f"alpha*={a_star:.2f}")


# tuning on raw RPS selects the same alpha as tuning on debiased RPS would: the
# debias offset is a per-item constant independent of the prediction. Pinned on
# variable-M panels, where the two scores differ by a non-constant offset.
rs5 = np.random.RandomState(5)
Mv = rs5.randint(2, 30, size=300)
pv5 = rs5.dirichlet(np.ones(5), size=300)
cv5 = np.stack([rs5.multinomial(int(m), p) for m, p in zip(Mv, pv5)]).astype(float)
mean5 = (pv5 * np.arange(5)).sum(1) + rs5.normal(0, 0.5, 300)
a_raw, _ = R.tune_alpha(mean5, cv5, 5)
deb = np.array([mt.rps_debiased(R.spike(mean5, 5, float(a)), cv5).mean() for a in R.ALPHAS])
check("tune_alpha on raw RPS == argmin of debiased RPS (variable M)",
      a_raw == float(R.ALPHAS[int(np.argmin(deb))]), f"alpha*={a_raw:.2f}")
c1 = np.eye(5)[rs5.randint(0, 5, 300)]
a1, _ = R.tune_alpha(mean5, c1, 5)
check("tune_alpha works at M = 1", 0.0 < a1 < 1.0, f"alpha*={a1:.2f}")


# --------------------------------------------------------------------------- #
print("\n5b. Rollout level alignment")
g = torch.Generator().manual_seed(31)
pm = torch.rand(50, 5, generator=g, dtype=torch.float64).numpy()
pm = pm / pm.sum(1, keepdims=True)
out, flipped = S2.align_rollout(pm, False)
check("reverse=False: rollout is passed through untouched", not flipped and out is pm)
out, flipped = S2.align_rollout(pm, True)
check("reverse=True: rollout is reversed", flipped and np.allclose(out, pm[:, ::-1]))
check("...and the result is still a pmf and C-contiguous",
      np.allclose(out.sum(1), 1.0) and out.flags["C_CONTIGUOUS"])
check("...reversing twice is the identity",
      np.array_equal(S2.align_rollout(out, True)[0], pm))

# the correction must NOT touch any trained arm: RPS and MAE are invariant when
# prediction and target reverse together, which is precisely why this bug hid
cnt = np.random.RandomState(3).multinomial(9, [.1, .2, .3, .2, .2], size=50).astype(float)
a_ = EV.arm_metrics(pm, cnt, n_boot=10)
b_ = EV.arm_metrics(np.ascontiguousarray(pm[:, ::-1]),
                    np.ascontiguousarray(cnt[:, ::-1]), n_boot=10)
check("joint reversal leaves RPS and MAE unchanged - why only the rollout broke",
      abs(a_["rps_debiased"] - b_["rps_debiased"]) < 1e-12
      and abs(a_["mae"] - b_["mae"]) < 1e-12,
      f"rps {a_['rps_debiased']:.6f} vs {b_['rps_debiased']:.6f}")
check("...while reversing only the prediction changes them a lot",
      abs(a_["rps_debiased"] - EV.arm_metrics(
          np.ascontiguousarray(pm[:, ::-1]), cnt, n_boot=10)["rps_debiased"]) > 1e-3)

# --------------------------------------------------------------------------- #
print("\n6. Metric wiring")
P_ = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
c_ = np.array([[2.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
m = EV.arm_metrics(P_, c_, n_boot=10)
check("mae is the mean |E[pred] - E[panel]|", abs(m["mae"] - 0.25) < 1e-12,
      f"mae={m['mae']}")
check("arm_metrics carries the full metrics.summary block",
      all(k in m for k in ("rps", "rps_debiased", "emd", "soft_ce", "cov90",
                           "ece_top", "ece_thr")))

m = EV.arm_metrics(np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]),
                   np.array([[2.0, 1.0, 0.0], [1.0, 4.0, 95.0]]), n_boot=10)
check("M-weighted and item-mean averages are genuinely different numbers",
      abs(m["mae"] - m["mae_Mw"]) > 1e-6
      and abs(m["rps_debiased"] - m["rps_debiased_Mw"]) > 1e-6,
      f"mae {m['mae']:.4f} vs {m['mae_Mw']:.4f}; "
      f"rps {m['rps_debiased']:.4f} vs {m['rps_debiased_Mw']:.4f}")

rng = np.random.RandomState(0)
p_ = np.array([0.1, 0.3, 0.6])
c_ = rng.multinomial(400, p_, size=300).astype(float)
m = EV.arm_metrics(np.tile(p_, (300, 1)), c_, n_boot=50)
check("a perfect prediction scores ~0 debiased RPS", abs(m["rps_debiased"]) < 5e-4,
      f"rps_db={m['rps_debiased']:+.2e}")
check("bootstrap SEs are finite and positive",
      m["rps_debiased_se"] > 0 and m["mae_se"] > 0)

# M = 1: the distributional estimators go to None, raw RPS and point metrics stay
y1 = np.random.RandomState(6).randint(0, 3, 200)
m1 = EV.arm_metrics(np.tile([0.2, 0.3, 0.5], (200, 1)), np.eye(3)[y1], n_boot=20,
                    n_boot_null=20)
check("M=1: regime is point and debiased RPS is None",
      m1["regime"] == "point" and m1["rps_debiased"] is None
      and m1["rps_debiased_se"] is None and m1["entropy_target_mm"] is None)
check("M=1: raw RPS, its SE and the point metrics are finite",
      all(np.isfinite(m1[k]) for k in ("rps", "rps_se", "acc", "mae_median", "qwk", "nll")))
mix = np.eye(3)[y1] + np.eye(3)[(y1 + 1) % 3] * (np.arange(200) % 2)[:, None]
mm = EV.arm_metrics(np.tile([0.2, 0.3, 0.5], (200, 1)), mix, n_boot=20, n_boot_null=20)
check("mixed M: debiased RPS is reported on the M>=2 subset, with its size",
      mm["regime"] == "point" and mm["n_M2subset"] == 100
      and np.isfinite(mm["rps_debiased_M2subset"]))


# --------------------------------------------------------------------------- #
print("\n4d. A whole task round-trips to disk")
_g = torch.Generator().manual_seed(77)
_n, _d, _K = 90, 5, 4
_X = torch.randn(_n, _d, generator=_g, dtype=torch.float64)
_w = torch.randn(_d, _K, generator=_g, dtype=torch.float64)
_p = torch.softmax(_X @ _w, dim=1)
_c = torch.stack([torch.tensor(np.random.RandomState(i).multinomial(4, _p[i].numpy()),
                               dtype=torch.float64) for i in range(_n)])
_split = ["train"] * 50 + ["val"] * 20 + ["test"] * 20


def _toy_file(path: Path, counts=_c, reverse=False):
    torch.save({"X": _X.float(), "counts": counts.long(), "K": _K, "split": _split,
                "item_id": [f"i{i}" for i in range(_n)],
                "group_id": [f"g{i}" for i in range(_n)],
                "task": "toy", "model": "toy/backbone", "reverse_rollout": reverse,
                "logit_pmf": _p.double()}, path)


_kw = dict(steps=120, select_steps=40, lr=3e-2, seed=0, n_boot_null=20)
with tempfile.TemporaryDirectory() as _td:
    _td = Path(_td)
    _toy_file(_td / "toy.pt")
    _rows = S2.run_task(_td / "toy.pt", "cpu", np.array([1e-2, 1.0]),
                        pred_dir=_td / "preds", **_kw)
    check("run_task returns every arm",
          {r["arm"] for r in _rows} == {"climatology", "climatology_itemavg",
                                        "logit_rollout", "spike_0.9", "spike_tuned",
                                        "softmax_map__ce", "softmax_map__rps",
                                        "ordinal_map__ce", "ordinal_map__rps"},
          f"{sorted({r['arm'] for r in _rows})}")
    try:
        _txt = _json.dumps(_rows)
        _enc, _why = True, ""
    except TypeError as e:
        _enc, _why = False, str(e)
    check("...and every value in them is JSON-encodable", _enc, _why)
    _blob = torch.load(_td / "preds" / "toy.pt", weights_only=False)
    check("the preds file carries val+test pmfs for every arm plus the ridge mean",
          set(_blob["pred_test"]) == {r["arm"] for r in _rows} | {"ridge_mean"}
          and all(_blob["pred_test"][a].shape == (20, _K) for a in EV.pmf_arms(_blob)))
    check("...with counts and item ids aligned to them",
          list(_blob["item_id_test"]) == [f"i{i}" for i in range(70, 90)]
          and np.array_equal(_blob["counts_test"], _c[70:].numpy()))
    check("rows re-scored from the preds file reproduce the reported metrics",
          all(abs(EV.arm_metrics(_blob["pred_test"][r["arm"]], _blob["counts_test"],
                                 n_boot_null=20)["rps"] - r["test"]["rps"]) < 1e-6
              for r in _rows))
    _toy_file(_td / "rev.pt", reverse=True)
    _rr = S2.run_task(_td / "rev.pt", "cpu", np.array([1.0]), pred_dir=_td / "preds",
                      heads=("softmax",), losses=("ce",), **_kw)
    _rb = torch.load(_td / "preds" / "rev.pt", weights_only=False)
    check("reverse_rollout in the activation file flips the rollout and is recorded",
          _rb["reverse_rollout"] is True
          and np.allclose(_rb["pred_test"]["logit_rollout"],
                          _p[70:].numpy()[:, ::-1], atol=1e-6))


print("\n4e. The training-size sweep down-samples train and NOTHING else")
# The sweep's whole claim is that the only thing differing between two fractions
# is how many training items there were. Three ways that can quietly stop being
# true, all checked here: the fractions stop being nested (so a difference is
# partly a different draw), val/test move with train (so the arms are no longer
# scored on the same items), or the default path stops being the identity (so
# the 100% cell, which is READ OFF THE COMMITTED RUN rather than refit, is no
# longer the same experiment as the down-sampled ones).
_tr = np.zeros(200, dtype=bool)
_tr[np.arange(0, 200, 2)] = True            # 100 training items, interleaved
_subs = {f: S2.subsample_train(_tr, f, 0) for f in (0.25, 0.5, 0.75, 1.0)}

check("frac=1.0 returns the mask untouched, so the default path is the identity",
      _subs[1.0] is _tr)
check("...and so does anything above it",
      S2.subsample_train(_tr, 1.0, 3) is _tr)
check("each fraction keeps round(frac * n_train) items",
      [int(_subs[f].sum()) for f in (0.25, 0.5, 0.75, 1.0)] == [25, 50, 75, 100],
      f"{[int(_subs[f].sum()) for f in (0.25, 0.5, 0.75, 1.0)]}")
check("every kept item was a training item to begin with",
      all(bool((_subs[f] & ~_tr).sum() == 0) for f in _subs))
check("the fractions are NESTED within a seed (25 < 50 < 75 < 100)",
      all(bool((_subs[a] & ~_subs[b]).sum() == 0)
          for a, b in [(0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]))
check("a different subsample seed draws a different subset",
      not np.array_equal(_subs[0.5], S2.subsample_train(_tr, 0.5, 1)))
check("...but the same seed is reproducible",
      np.array_equal(_subs[0.5], S2.subsample_train(_tr, 0.5, 0)))
check("the subsample of a mask cannot touch val or test",
      # the complement of `tr` is untouched by construction: the returned mask is
      # built from zeros and only train indices are ever set
      all(bool((_subs[f] | _tr).sum() == _tr.sum()) for f in _subs))
check("a fraction that would leave zero training items is refused",
      _raises(lambda: S2.subsample_train(_tr, 1e-4, 0)))
check("a nonsensical fraction is refused rather than silently clamped",
      _raises(lambda: S2.subsample_train(_tr, 0.0, 0))
      and _raises(lambda: S2.subsample_train(_tr, -0.5, 0)))


# --------------------------------------------------------------------------- #
# 4f. Domain holdout (stage2.holdout_split), the covariate-shift arm
#
# One fit has to yield BOTH blocks - the held-out domain and the original
# in-domain test items - or the two are scored by different heads and any
# difference between them is partly a difference between fits.
# --------------------------------------------------------------------------- #
print("\n4f. Domain holdout: split construction")

_nf = 200
_grp = np.array([f"g{i % 5}" for i in range(_nf)])
_rs = np.random.RandomState(4)
_lab = _rs.choice(["train", "val", "test"], size=_nf, p=[0.6, 0.2, 0.2])
_tr0, _va0, _te0 = _lab == "train", _lab == "val", _lab == "test"
_tr, _va, _te, _isd = S2.holdout_split(_tr0, _va0, _te0, _grp, ["g3"])

check("4f: the held-out domain is exactly the matching groups",
      np.array_equal(_isd, _grp == "g3"))
check("4f: the held-out domain is gone from BOTH train and val",
      not (_tr & _isd).any() and not (_va & _isd).any())
check("4f: ...which matters because lambda is selected on val - a held-out "
      "domain left in val is a domain the fit has effectively seen",
      (_va0 & _isd).sum() > 0 and (_va & _isd).sum() == 0)
check("4f: every held-out item lands in test",
      (_te & _isd).sum() == _isd.sum())
check("4f: the original in-domain test items are still test, so one fit gives "
      "both blocks",
      ((_te0 & ~_isd) == (_te & ~_isd)).all() and (_te & ~_isd).sum() > 0)
check("4f: nothing outside the held-out domain moves",
      np.array_equal(_tr[~_isd], _tr0[~_isd])
      and np.array_equal(_va[~_isd], _va0[~_isd]))
check("4f: the three masks stay a partition of the corpus",
      (_tr.astype(int) + _va.astype(int) + _te.astype(int) <= 1).all()
      and (_tr | _va | _te).sum() == (_tr0 | _va0 | _te0).sum())
check("4f: an empty group list is rejected",
      _raises(lambda: S2.holdout_split(_tr0, _va0, _te0, _grp, [])))
check("4f: a group list matching nothing is rejected, not silently a no-op",
      _raises(lambda: S2.holdout_split(_tr0, _va0, _te0, _grp, ["nope"])))
check("4f: holding out every group is rejected (empty train)",
      _raises(lambda: S2.holdout_split(_tr0, _va0, _te0, _grp,
                                       [f"g{i}" for i in range(5)])))
# the holdout is applied BEFORE subsample_train, so --train-frac keeps meaning
# "a fraction of what this arm trains on"
_half = S2.subsample_train(_tr, 0.5, 0)
check("4f: --train-frac composes with the holdout and never re-admits it",
      _half.sum() == int(round(0.5 * _tr.sum())) and not (_half & _isd).any())

# --------------------------------------------------------------------------- #
print("\n4g. brier and tbce heads through run_task")
with tempfile.TemporaryDirectory() as _tdn:
    _tdn = Path(_tdn)
    _toy_file(_tdn / "toy.pt")
    _rn = S2.run_task(_tdn / "toy.pt", "cpu", np.array([1e-2, 1.0]),
                      pred_dir=_tdn / "preds", heads=("softmax", "clm"),
                      losses=("brier", "tbce"), **_kw)
    _arms = {r["arm"] for r in _rn}
    check("brier and tbce each get a MAP arm per head",
          {"softmax_map__brier", "ordinal_map__brier",
           "softmax_map__tbce", "ordinal_map__tbce"} <= _arms, f"{sorted(_arms)}")
    check("...and the task JSON-encodes", _json.loads(_json.dumps(_rn))[0]["task"] == "toy")


# --------------------------------------------------------------------------- #
print("\n4h. fit_split=val swaps train<->val; fit_n takes an absolute count")
with tempfile.TemporaryDirectory() as _tdv:
    _tdv = Path(_tdv)
    _toy_file(_tdv / "toy.pt")
    _kh = dict(_kw, heads=("softmax",), losses=("ce",))
    _rv = S2.run_task(_tdv / "toy.pt", "cpu", np.array([1.0]), pred_dir=_tdv / "p1",
                      fit_split="val", **_kh)
    _hv = next(r for r in _rv if r["arm"] == "softmax_map__ce")
    check("fit_split=val: head trains on val, train becomes the tuning set, test unchanged",
          (_hv["n_train"], _hv["n_val"], _hv["n_test"]) == (20, 50, 20),
          f"{(_hv['n_train'], _hv['n_val'], _hv['n_test'])}")
    _rn2 = S2.run_task(_tdv / "toy.pt", "cpu", np.array([1.0]), pred_dir=_tdv / "p2",
                       fit_split="val", fit_n=12, train_subsample_seed=1, **_kh)
    _hn = next(r for r in _rn2 if r["arm"] == "softmax_map__ce")
    check("fit_n=12 draws exactly 12 items", _hn["n_train"] == 12)
    check("fit_n outside 1..n is refused",
          _raises(lambda: S2.run_task(_tdv / "toy.pt", "cpu", np.array([1.0]),
                                      pred_dir=_tdv / "p3", fit_split="val",
                                      fit_n=999, **_kh)))


# --------------------------------------------------------------------------- #
print("\n7. M = 1 end to end: train_heads -> verbal attach -> recalibrate -> evaluate")
_y = torch.tensor([int(np.random.RandomState(100 + i).choice(_K, p=_p[i].numpy()))
                   for i in range(_n)])
_c1 = torch.nn.functional.one_hot(_y, _K).double()
with tempfile.TemporaryDirectory() as _t1:
    _t1 = Path(_t1)
    _toy_file(_t1 / "toy1.pt", counts=_c1)
    _r1 = S2.run_task(_t1 / "toy1.pt", "cpu", np.array([1e-2, 1.0]),
                      pred_dir=_t1 / "preds", **_kw)
    check("M=1 run_task completes with every arm",
          len({r["arm"] for r in _r1}) == 9)
    check("...its rows report the point regime and JSON-encode",
          all(r["test"]["regime"] == "point" and r["test"]["rps_debiased"] is None
              for r in _r1) and bool(_json.dumps(_r1)))
    # a verbal run writes one row per item of the split, keyed by dataset name
    for s, sl in (("val", slice(50, 70)), ("test", slice(70, 90))):
        ids = [f"i{i}" for i in range(_n)][sl][::-1]            # deliberately reordered
        pm = np.full((20, _K), 1.0 / _K)
        pm[:, 0] += np.linspace(0, 0.5, 20)
        pm /= pm.sum(1, keepdims=True)
        _json.dump({"toy1": {"item_id": ids, "pmf": pm.tolist(), "K": _K,
                             "degenerate_rate": 0.0}}, open(_t1 / f"verbal_{s}.json", "w"))
    b1 = EV.load_preds(_t1 / "preds" / "toy1.pt")
    check("attach_verbal joins verbal onto the split by item id, reordering",
          EV.attach_verbal(b1, _t1) and b1["pred_test"]["verbal"].shape == (20, _K)
          and np.allclose(b1["pred_test"]["verbal"][-1],
                          np.array(pm[0], dtype=np.float32), atol=1e-6))
    bad = EV.load_preds(_t1 / "preds" / "toy1.pt")
    bad["item_id_test"] = np.array(["nope"] * 20)
    check("...and refuses a verbal file that does not cover the split",
          _raises(lambda: EV.attach_verbal(bad, _t1)))
    new, recs = RC.recalibrate_blob(b1, "pmf", 1, "ce")
    _fitted = [a for a in EV.pmf_arms(b1) if not a.startswith("climatology")]
    check("recalibrate_blob adds '+ts' twins for every non-constant arm, keeping the raw ones",
          {a + "+ts" for a in _fitted} == set(new["pred_test"]) - set(b1["pred_test"])
          and set(EV.pmf_arms(b1)) <= set(new["pred_test"]))
    check("...each a valid pmf", all(np.allclose(new["pred_test"][a].sum(1), 1, atol=1e-5)
                                     for a in EV.pmf_arms(new)))
    check("...with a temperature record per arm that JSON-encodes",
          len(recs) == len(_fitted) and all("temperature" in r for r in recs)
          and bool(_json.dumps(recs)))
    check("recalibration never looks at test: refit with shuffled test pmfs, same theta",
          RC.recalibrate_blob({**b1, "pred_test": {a: v[::-1].copy() for a, v in
                                                   b1["pred_test"].items()}},
                              "pmf", 1, "ce")[1][0]["theta"] == recs[0]["theta"])
    (_t1 / "preds_recal").mkdir()
    torch.save(new, _t1 / "preds_recal" / "toy1.pt")
    ev = EV.evaluate_file(_t1 / "preds_recal" / "toy1.pt", _t1, "test", n_boot_null=20)
    check("evaluate_file scores raw, recalibrated and verbal arms",
          {"verbal", "verbal+ts", "softmax_map__ce+ts"} <= {r["arm"] for r in ev})
    md = EV.to_markdown(ev, "test")
    check("...and the markdown uses the point-metric headline at M = 1",
          "point metrics" in md and "QWK" in md and "RPS(deb)" not in md)
    check("...and JSON-encodes", bool(_json.dumps(ev)))


# --------------------------------------------------------------------------- #
def test_heads():
    assert not FAIL, FAIL


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        sys.exit(1)
    print("head training, baselines, scoring and recalibration verified")
