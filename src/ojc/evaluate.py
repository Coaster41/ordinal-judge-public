"""Score saved predictions: every arm of every preds file, plus verbal.

    python -m ojc.evaluate --run-dir runs/<slug>                 # preds/
    python -m ojc.evaluate --run-dir runs/<slug> --preds-subdir preds_recal

Reads the preds files written by `ojc.train_heads` (or `ojc.recalibrate`, which
writes the same schema), attaches the verbal-elicitation pmfs from
`<run-dir>/verbal_{val,test}.json` when they exist, and scores each arm on the
chosen split. Writes `<run-dir>/eval<_suffix>.json` (every metric) and
`eval<_suffix>.md` (the headline table) and prints the table.

Which metrics are the headline depends on the panel size (`metrics.regime`):

  distributional  every item has M >= 2 ratings. Read debiased RPS (the
                  finite-M-unbiased estimate of the population RPS), MAE of
                  the predicted mean, and ece_thr against its null floor.
  point           some item has a single rating. Debiased RPS is undefined;
                  read accuracy, MAE of the predicted median, QWK, Spearman,
                  NLL and raw RPS. If some items do have M >= 2, debiased RPS
                  is also reported on that subset, with its size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ojc import metrics as mt

NON_PMF_ARMS = {"ridge_mean"}

HEADLINE = {
    "distributional": [("rps_debiased", "RPS(deb)"), ("mae", "MAE"),
                       ("ece_thr", "ECE_thr"), ("ece_thr_above_floor", "ECE_thr-floor"),
                       ("ece_top", "ECE_top"), ("soft_ce", "CE"), ("acc", "acc")],
    "point": [("acc", "acc"), ("mae_median", "MAE(med)"), ("qwk", "QWK"),
              ("spearman", "Spearman"), ("nll", "NLL"), ("rps", "RPS"),
              ("ece_top", "ECE_top")],
}


# --------------------------------------------------------------------------- #
# per-arm metrics
# --------------------------------------------------------------------------- #

def arm_metrics(P: np.ndarray, counts: np.ndarray, n_boot: int = 500,
                seed: int = 0, n_boot_null: int = 200) -> dict:
    """Every scalar we might want, plus item-bootstrap SEs on the headline RPS
    and MAE. Per-item predictions are saved alongside, so anything not here can
    be added later without refitting.

    `rps_debiased*` are None unless every item has M >= 2 (see module
    docstring); raw `rps*` are always defined.
    """
    P = np.asarray(P, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    K = P.shape[1]
    lv = np.arange(K, dtype=np.float64)
    phat = mt.to_pmf(counts)
    M = counts.sum(-1)
    mu_p = (P * lv).sum(1)
    mu_e = (phat * lv).sum(1)
    dist = mt.regime(counts) == "distributional"

    rps_raw_i = mt.rps(P, counts)
    rps_deb_i = mt.rps_debiased(P, counts) if dist else None
    mae_i = np.abs(mu_p - mu_e)
    out = {
        **mt.summary(P, counts, n_boot_null=n_boot_null, seed=seed),
        "mae": float(mae_i.mean()),
        "mae_annotator": float((phat * np.abs(lv[None, :] - mu_p[:, None])).sum(1).mean()),
        # M-weighted variants: the losses are M-weighted, so these are the
        # per-ANNOTATION versions of the same numbers
        "rps_Mw": float((rps_raw_i * M).sum() / M.sum()),
        "rps_debiased_Mw": float((rps_deb_i * M).sum() / M.sum()) if dist else None,
        "mae_Mw": float((mae_i * M).sum() / M.sum()),
        "mean_pred": float(mu_p.mean()),
        "mean_target": float(mu_e.mean()),
        "entropy_pred": float(mt.entropy(P).mean()),
        "entropy_target_mm": float(mt.entropy_miller_madow(counts).mean()) if dist else None,
        "n_items": int(len(P)),
        "M_min": float(M.min()), "M_max": float(M.max()),
    }
    if not dist and (M >= 2).any():
        sub = M >= 2
        out["rps_debiased_M2subset"] = float(mt.rps_debiased(P[sub], counts[sub]).mean())
        out["n_M2subset"] = int(sub.sum())
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(P), size=(n_boot, len(P)))
    out["rps_se"] = float(np.std(rps_raw_i[idx].mean(1), ddof=1))
    out["rps_debiased_se"] = float(np.std(rps_deb_i[idx].mean(1), ddof=1)) if dist else None
    out["mae_se"] = float(np.std(mae_i[idx].mean(1), ddof=1))
    return out


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_preds(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def pmf_arms(blob: dict) -> list[str]:
    return [a for a in blob["pred_test"] if a not in NON_PMF_ARMS]


def _join(v: dict, item_ids: np.ndarray, where: str) -> np.ndarray:
    """Reorder a verbal entry's pmf rows onto `item_ids`. Every item must be
    present - verbal.py writes one row per item of the split, substituting the
    train marginal for a degenerate decode, so a missing item means the verbal
    run and the preds file disagree about the split."""
    pos = {iid: i for i, iid in enumerate(v["item_id"])}
    missing = [iid for iid in item_ids if iid not in pos]
    if missing:
        raise ValueError(f"{where}: {len(missing)} of {len(item_ids)} items have no "
                         f"verbal prediction (e.g. {missing[:3]}) - was verbal run "
                         f"on the same dataset/split?")
    p = np.asarray(v["pmf"], dtype=np.float64)[[pos[i] for i in item_ids]]
    if not np.isfinite(p).all():
        raise ValueError(f"{where}: non-finite verbal pmf rows")
    return p / p.sum(-1, keepdims=True)


def attach_verbal(blob: dict, run_dir: Path, verbose: bool = True) -> bool:
    """Add a `verbal` arm to a preds blob (in place) from
    `<run_dir>/verbal_{val,test}.json`. Returns False, and changes nothing, if
    either file is missing or has no entry for this task.

    The verbal prompt uses the same level direction as the logit prompt, so
    it is reversed exactly when the rollout was (`reverse_rollout`).
    """
    files = {s: run_dir / f"verbal_{s}.json" for s in ("val", "test")}
    if not all(f.exists() for f in files.values()):
        return False
    key = blob["file"]
    js = {s: json.load(open(f)) for s, f in files.items()}
    if not all(key in j for j in js.values()):
        if verbose:
            print(f"  [{key}] no verbal entry in {files['val'].name}/{files['test'].name}")
        return False
    for s in ("val", "test"):
        p = _join(js[s][key], np.asarray(blob[f"item_id_{s}"]), f"verbal_{s}[{key}]")
        if blob.get("reverse_rollout"):
            p = np.ascontiguousarray(p[:, ::-1])
        blob[f"pred_{s}"]["verbal"] = p.astype(np.float32)
    blob.setdefault("verbal_degenerate_rate",
                    {s: float(js[s][key].get("degenerate_rate", 0.0)) for s in js})
    return True


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def _fmt(v, se=None) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    s = f"{v:.4f}"
    return f"{s} ± {se:.4f}" if se is not None else s


def to_markdown(rows: list[dict], split: str) -> str:
    md = []
    for f in dict.fromkeys(r["file"] for r in rows):
        rs = [r for r in rows if r["file"] == f]
        m = rs[0]["metrics"]
        reg = m["regime"]
        md.append(f"## {f}  ({split}, n={m['n_items']}, K={rs[0]['K']}, "
                  f"M in [{m['M_min']:.0f}, {m['M_max']:.0f}] -> {reg} metrics)\n")
        cols = HEADLINE[reg]
        extra = "rps_debiased_M2subset" in m
        hdr = ["arm"] + [c[1] for c in cols] + (["RPS(deb, M>=2)"] if extra else [])
        md.append("| " + " | ".join(hdr) + " |")
        md.append("|" + "---|" * len(hdr))
        for r in rs:
            mm = r["metrics"]
            cells = [r["arm"]]
            for key, _ in cols:
                se = mm.get(f"{key}_se") if key in ("rps_debiased", "rps", "mae") else None
                cells.append(_fmt(mm.get(key), se))
            if extra:
                cells.append(_fmt(mm.get("rps_debiased_M2subset")))
            md.append("| " + " | ".join(cells) + " |")
        if extra:
            md.append(f"\nRPS(deb, M>=2) is on the {m['n_M2subset']} items with at "
                      f"least two ratings.")
        md.append("")
    md.append("Lower is better except acc, QWK, Spearman.")
    if any(r["metrics"]["regime"] == "distributional" for r in rows):
        md[-1] += (" `ECE_thr-floor` is ECE_thr minus the expected ECE_thr of a "
                   "perfectly calibrated model on the same items; ~0 means no "
                   "detectable miscalibration.")
    return "\n".join(md)


def evaluate_file(path: Path, run_dir: Path | None, split: str, arms=None,
                  n_boot_null: int = 200) -> list[dict]:
    blob = load_preds(path)
    if run_dir is not None and "verbal" not in blob["pred_test"]:
        attach_verbal(blob, run_dir)
    counts = np.asarray(blob[f"counts_{split}"], dtype=np.float64)
    rows = []
    for arm in pmf_arms(blob):
        if arms and arm not in arms:
            continue
        P = np.asarray(blob[f"pred_{split}"][arm], dtype=np.float64)
        rows.append({"backbone": blob["backbone"], "task": blob["task"],
                     "file": blob["file"], "K": int(blob["K"]), "split": split,
                     "arm": arm, "source": str(path),
                     "metrics": arm_metrics(P, counts, n_boot_null=n_boot_null)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, help="runs/<slug>")
    ap.add_argument("--preds-subdir", default="preds",
                    help="preds (default), preds_recal, or a tagged preds_<tag>")
    ap.add_argument("--tasks", nargs="*", default=None, help="restrict to these file stems")
    ap.add_argument("--arms", nargs="*", default=None, help="restrict to these arms")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--no-verbal", action="store_true",
                    help="do not attach verbal_{val,test}.json")
    ap.add_argument("--n-boot-null", type=int, default=200)
    ap.add_argument("--out", default=None,
                    help="output stem (default <run-dir>/eval[_<subdir suffix>])")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    pdir = run_dir / a.preds_subdir
    files = sorted(pdir.glob("*.pt"))
    if a.tasks:
        files = [f for f in files if f.stem in a.tasks]
    if not files:
        raise SystemExit(f"no preds files in {pdir}")

    rows = []
    for f in files:
        rows += evaluate_file(f, None if a.no_verbal else run_dir, a.split,
                              arms=a.arms, n_boot_null=a.n_boot_null)

    suffix = a.preds_subdir[len("preds"):] if a.preds_subdir.startswith("preds") else f"_{a.preds_subdir}"
    stem = Path(a.out) if a.out else run_dir / f"eval{suffix}{'' if a.split == 'test' else '_val'}"
    md = to_markdown(rows, a.split)
    json.dump(rows, open(stem.with_suffix(".json"), "w"), indent=2)
    stem.with_suffix(".md").write_text(md + "\n")
    print(md)
    print(f"\nwrote {stem.with_suffix('.json')} and {stem.with_suffix('.md')}")


if __name__ == "__main__":
    main()
