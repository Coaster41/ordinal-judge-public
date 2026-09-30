"""One forward pass per item -> last-layer activation AND logit-rollout pmf.

    python -m ojc.extract --model Qwen/Qwen3.5-0.8B --data examples/hs2_helpfulness.jsonl

The logit rollout already runs a forward pass whose final hidden state is exactly
the feature a head trains on, so both come out of a single pass: the heads are
trained on the very representation the rollout scored, with no chance of the two
drifting apart in prompt, batch composition or thinking mode.

  * Logits come from the model's own head (`backbone.forward_logits`), so any
    logit soft-capping the model does is applied.
  * Exactly one layer is captured, by a hook on the final norm.
  * Length-sorted batching cuts padding; rows are scattered back to dataset
    order by their original indices before saving (tests/test_extract.py pins
    the round trip - a wrong inverse permutation would attach every activation
    to the wrong item and nothing downstream would notice).

bf16 forward passes are NOT batch-invariant: the same item's activation moves by
a small fraction of a feature SD with batch composition. Keep --batch-size fixed
across any runs you intend to compare, and when resuming a partial extraction.

Writes `<out>/<task>.pt` (default out: runs/<model-slug>/activations/), verified
against a fresh load of the dataset before it replaces any existing file.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from ojc import backbone as B
from ojc import data as D
from ojc import prompts as P


def sorted_batches(lengths: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Indices grouped into batches of similar token length.

    Returns batches of ORIGINAL indices; the caller scatters results back by
    those indices, so output order never depends on processing order.
    """
    order = np.argsort(lengths, kind="stable")
    return [order[i: i + batch_size] for i in range(0, len(order), batch_size)]


def _source_meta(source: str, spec: str | None, data_dir: Path, n_max) -> dict:
    """Enough to reload exactly this dataset later (see `verify`)."""
    if D.is_jsonl(source):
        sp = Path(spec) if spec else D.spec_path_for(Path(source))
        return {"source": str(Path(source).resolve()), "spec": str(sp.resolve()),
                "data_dir": None, "n_max": None}
    return {"source": source, "spec": None, "data_dir": str(Path(data_dir).resolve()),
            "n_max": n_max}


def reload_dataset(blob: dict) -> D.OrdinalDataset:
    return D.resolve(blob["source"], blob.get("spec"),
                     Path(blob["data_dir"]) if blob.get("data_dir") else D.DEFAULT_DATA_DIR,
                     blob.get("n_max"))


@torch.no_grad()
def run_task(ds: D.OrdinalDataset, stem: str, meta: dict, model, tok, text_model,
             hook: B.FinalNormHook, batch_size: int, max_len: int, out_dir: Path,
             report_every: int = 25) -> Path:
    K, n = ds.K, len(ds)
    hidden = B.hidden_size(model, text_model)
    dev = B.input_device(model)
    opt_ids = P.option_token_ids(tok, K)

    print(f"[{stem}] n={n} K={K} M in [{ds.M.min()},{ds.M.max()}] hidden={hidden}",
          flush=True)
    prompts = [P.build_logit_prompt(tok, ds.text[i], ds.level_labels, max_len) for i in range(n)]
    lengths = np.array([len(P.encode(tok, p)) for p in prompts])
    print(f"[{stem}] tokens: median {int(np.median(lengths))} mean {lengths.mean():.0f} "
          f"max {lengths.max()} (>{max_len}: {int((lengths > max_len).sum())})", flush=True)

    X = np.zeros((n, hidden), dtype=np.float32)
    pmf = np.zeros((n, K), dtype=np.float64)
    filled = np.zeros(n, dtype=bool)

    t0 = time.time()
    for bi, idx in enumerate(sorted_batches(lengths, batch_size)):
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len, add_special_tokens=False).to(dev)
        hook.reset()
        out = B.forward_logits(model, enc)
        if hook.n_calls != 1:
            raise RuntimeError(f"final-norm hook fired {hook.n_calls} times, expected 1")

        X[idx] = hook.h.numpy()
        lg = out.logits[:, -1, :].float()
        pmf[idx] = torch.softmax(lg[:, opt_ids.to(lg.device)], -1).double().cpu().numpy()
        filled[idx] = True
        del out, enc

        if bi % report_every == 0:
            done = int(filled.sum())
            print(f"  {done}/{n}  {done / max(time.time() - t0, 1e-9):.2f} items/s", flush=True)

    if not filled.all():
        raise RuntimeError(f"{stem}: {int((~filled).sum())} items never filled")
    if not np.isfinite(X).all():
        raise RuntimeError(f"{stem}: non-finite activations")
    if not np.isfinite(pmf).all() or not np.allclose(pmf.sum(1), 1.0, atol=1e-6):
        raise RuntimeError(f"{stem}: logit pmf rows do not sum to 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "X": torch.from_numpy(X),
        "logit_pmf": torch.from_numpy(pmf),
        "counts": torch.tensor(ds.counts),
        "split": list(ds.split), "item_id": list(ds.item_id),
        "group_id": list(ds.group_id),
        "K": K, "task": stem, "level_labels": list(ds.level_labels),
        "reverse_rollout": bool(ds.reverse_rollout),
        **meta,
        "thinking": "off", "batch_size": batch_size, "max_len": max_len,
        "length_sorted": True, "prompt": "logit", "prefill": P.PREFILL,
        "model": getattr(model.config, "_name_or_path", "unknown"),
    }, tmp)
    verify(tmp)
    os.replace(tmp, path)
    el = time.time() - t0
    print(f"[{stem}] saved {path}  X={X.shape}  {el / 60:.1f} min ({n / el:.2f} items/s)",
          flush=True)
    return path


def verify(path: Path) -> bool:
    """Confirm a saved file matches a freshly loaded dataset, row for row.

    This checks ids, counts and splits against the source; it does NOT prove
    each activation row belongs to its item (that is the permutation test's
    job, in tests/test_extract.py).
    """
    d = torch.load(path, weights_only=False)
    ds = reload_dataset(d)
    X, pmf = d["X"], d["logit_pmf"]
    name = Path(path).name
    if X.shape[0] != len(ds):
        raise ValueError(f"{name}: {X.shape[0]} rows, dataset has {len(ds)}")
    if pmf.shape != (len(ds), ds.K):
        raise ValueError(f"{name}: pmf {tuple(pmf.shape)} != {(len(ds), ds.K)}")
    if list(d["item_id"]) != list(ds.item_id):
        raise ValueError(f"{name}: item ids do not match a fresh dataset load")
    if not np.array_equal(d["counts"].numpy(), ds.counts):
        raise ValueError(f"{name}: annotator counts do not match")
    if list(d["split"]) != list(ds.split):
        raise ValueError(f"{name}: splits do not match")
    if not torch.isfinite(X).all():
        raise ValueError(f"{name}: non-finite activations")
    return True


def parse_n_max(s):
    return None if s is None or str(s).lower() in ("none", "all", "full") else int(s)


def add_data_args(ap: argparse.ArgumentParser):
    ap.add_argument("--data", nargs="+", required=True,
                    help="datasets: .jsonl paths (spec beside each as <name>.spec.json) "
                         f"and/or paper task names {D.PAPER_TASKS}")
    ap.add_argument("--spec", default=None,
                    help="spec file to use for EVERY .jsonl in --data (default: sidecar)")
    ap.add_argument("--data-dir", default=str(D.DEFAULT_DATA_DIR),
                    help="where the paper corpora's raw files live")
    ap.add_argument("--n-max", default="none",
                    help="subsample paper corpora to this many items ('none' = all)")


def add_model_args(ap: argparse.ArgumentParser):
    ap.add_argument("--model", required=True, help="Hugging Face model id or local path")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="keep FIXED across runs you compare (bf16 is not batch-invariant)")
    ap.add_argument("--max-len", type=int, default=2048,
                    help="prompt token budget; items are shortened in the middle to fit")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--cache-dir", default=None, help="HF cache dir (default: HF's own)")
    ap.add_argument("--run-root", default="runs", help="outputs go to <run-root>/<model-slug>/")


def load_all(a) -> list[tuple[str, D.OrdinalDataset, dict]]:
    """Resolve and sanity-check every dataset BEFORE the model is loaded, so a
    bad file fails in seconds rather than after the weights are on the GPU."""
    n_max = parse_n_max(a.n_max)
    out = []
    for src in a.data:
        ds = D.resolve(src, a.spec, Path(a.data_dir), n_max)
        if len(ds) == 0 or ds.M.min() < 1:
            raise SystemExit(f"{src}: empty dataset or items with no ratings")
        for s in D.SPLITS:
            if not (ds.split == s).any():
                raise SystemExit(f"{src}: the {s} split is empty")
        out.append((D.task_stem(src), ds, _source_meta(src, a.spec, Path(a.data_dir), n_max)))
        print(f"  [{out[-1][0]}] n={len(ds)} K={ds.K} M in [{ds.M.min()},{ds.M.max()}] "
              f"splits " + "/".join(str(int((ds.split == s).sum())) for s in D.SPLITS),
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_data_args(ap)
    add_model_args(ap)
    ap.add_argument("--out", default=None,
                    help="default <run-root>/<model-slug>/activations/")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip datasets whose file exists and verifies")
    a = ap.parse_args()

    out_dir = Path(a.out) if a.out else Path(a.run_root) / B.model_slug(a.model) / "activations"
    print(f"model={a.model} batch={a.batch_size} max_len={a.max_len} -> {out_dir}", flush=True)
    B.check_attention_indexing(a.model, a.batch_size, a.max_len, cache_dir=a.cache_dir)

    todo = []
    for stem, ds, meta in load_all(a):
        fp = out_dir / f"{stem}.pt"
        if a.skip_existing and fp.exists():
            try:
                verify(fp)
                print(f"  [{stem}] already done and verified, skipping", flush=True)
                continue
            except Exception as e:
                print(f"  [{stem}] existing file INVALID ({e}); redoing", flush=True)
        todo.append((stem, ds, meta))
    for stale in out_dir.glob("*.pt.tmp"):
        stale.unlink()
    if not todo:
        print("nothing to do")
        return

    model, tok, text_model = B.load_backbone(a.model, a.dtype, a.cache_dir)
    P.option_token_ids(tok, 10)  # fail now, not after the first dataset
    norm_mod, norm_name = B.final_norm_module(text_model)
    print(f"hooking final norm: {norm_name} ({type(norm_mod).__name__})", flush=True)
    hook = B.FinalNormHook(norm_mod)
    try:
        for stem, ds, meta in todo:
            run_task(ds, stem, meta, model, tok, text_model, hook, a.batch_size,
                     a.max_len, out_dir)
    finally:
        hook.close()


if __name__ == "__main__":
    main()
