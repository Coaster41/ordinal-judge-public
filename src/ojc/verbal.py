"""Verbalized distribution: ask the model how 100 annotators would split.

    python -m ojc.verbal --model Qwen/Qwen3.5-0.8B --data examples/hs2_helpfulness.jsonl

The prompt is the item text plus an instruction to report, for each of the K
options, how many of 100 independent annotators would choose it - as K
zero-padded 2-digit numbers back to back ("0515403010" for K = 5), after a
fixed worked example (`prompts.generic_example_digits`).

Decoding is a HAND-MASKED greedy decode: exactly 2K steps, each restricted to
the ten bare-digit tokens. Position alone fixes each digit's place value, so no
delimiter is needed and the output is structurally guaranteed to parse - there
is no regex and no parse failure. The one degenerate outcome is an all-zero
row; it is replaced by the training-split marginal and counted in
`degenerate_rate`.

Items are grouped into EXACT-length buckets, so every batch runs with no
padding at all.

`example_copy_rate` counts rows that reproduce the worked example verbatim.
Small models do this often, which makes the verbal arm uninformative for
them; it is reported, not corrected.

Run it on both val and test (the default): val is what `ojc.recalibrate` fits
the verbal arm's temperature on. Writes `<run-root>/<model-slug>/verbal_<split>.json`,
keyed by dataset name, which `ojc.evaluate` and `ojc.recalibrate` pick up
automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ojc import backbone as B
from ojc import metrics as mt
from ojc import prompts as P
from ojc.extract import add_data_args, add_model_args, load_all


@torch.no_grad()
def decode_digits(model, init_logits, cache, attn_mask, n_steps: int,
                  digit_ids10) -> np.ndarray:
    """Greedy-decode exactly n_steps digit tokens, masked to 0..9 at every step.

    `digit_ids10[i]` is the token id for digit i, so the argmax INDEX already
    equals the digit VALUE - no detokenisation needed.
    """
    device = init_logits.device
    batch = init_logits.shape[0]
    digit_ids10 = digit_ids10.to(device)
    digits = torch.zeros(batch, n_steps, dtype=torch.long, device=device)
    logits, am = init_logits, attn_mask
    for step in range(n_steps):
        choice = logits[:, digit_ids10].argmax(-1)   # index == digit value (0..9)
        digits[:, step] = choice
        if step == n_steps - 1:
            break
        next_ids = digit_ids10[choice].unsqueeze(1).to(am.device)
        am = torch.cat([am, torch.ones(batch, 1, dtype=am.dtype, device=am.device)], dim=1)
        out = B.forward_logits(model, {"input_ids": next_ids, "attention_mask": am},
                               use_cache=True, past_key_values=cache)
        logits = out.logits[:, -1, :].float()
        cache = out.past_key_values
    return digits.cpu().numpy()


def combine_digit_groups(raw: np.ndarray, K: int) -> np.ndarray:
    """(n, K * DIGITS_PER_LEVEL) single digits, most-significant first per group
    -> (n, K) counts in 0..N_ANNOTATORS-1."""
    g = raw.reshape(raw.shape[0], K, P.DIGITS_PER_LEVEL)
    place = 10 ** np.arange(P.DIGITS_PER_LEVEL - 1, -1, -1)
    return (g * place).sum(-1)


def digits_to_pmf(counts: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalise counts to a pmf; an all-zero row becomes `fallback`."""
    s = counts.sum(1, keepdims=True).astype(np.float64)
    degenerate = (s[:, 0] == 0)
    pmf = np.divide(counts, np.where(s == 0, 1, s), dtype=np.float64)
    pmf[degenerate] = fallback
    return pmf, degenerate


def save_result(path: Path, key: str, ds, pmf: np.ndarray, extra: dict):
    """One JSON per (model, split), keyed by dataset; rewritten atomically."""
    if not np.isfinite(pmf).all() or not np.allclose(pmf.sum(1), 1.0, atol=1e-6):
        raise RuntimeError(f"{key}: verbal pmf rows do not sum to 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    store = json.load(open(path)) if path.exists() else {}
    store[key] = {"item_id": list(map(str, ds.item_id)), "split": list(ds.split),
                  "K": int(ds.K), "pmf": pmf.tolist(), **extra}
    tmp = path.with_suffix(".json.tmp")
    json.dump(store, open(tmp, "w"))
    os.replace(tmp, path)
    print(f"  [{key}] saved {path}  degenerate {extra['degenerate_rate']:.1%}  "
          f"copied the worked example {extra.get('example_copy_rate', 0):.1%}", flush=True)


@torch.no_grad()
def run_task(full, key: str, split: str, model, tok, batch_size: int, max_len: int,
             path: Path, report_every: int = 25):
    marg = mt.to_pmf(full.counts[full.split == "train"]).mean(0)
    ds = full.subset(split)
    K, n = ds.K, len(ds)
    print(f"[{key}/{split}] verbal n={n} K={K}", flush=True)

    prompts = [P.build_verbal_prompt(tok, ds.text[i], K, max_len) for i in range(n)]
    ids_list = [P.encode(tok, p) for p in prompts]
    lengths = np.array([len(x) for x in ids_list])
    print(f"[{key}/{split}] tokens: median {int(np.median(lengths))} "
          f"max {lengths.max()}", flush=True)

    pmf = np.zeros((n, K), dtype=np.float64)
    copied = np.zeros(n, dtype=bool)
    degenerate = np.zeros(n, dtype=bool)
    filled = np.zeros(n, dtype=bool)
    digit_ids10 = P.option_token_ids(tok, 10)
    example = P.generic_example_digits(K)

    buckets: dict[int, list[int]] = {}
    for i, L in enumerate(lengths):
        buckets.setdefault(int(L), []).append(i)
    batches = [np.array(g[j: j + batch_size])
               for g in buckets.values() for j in range(0, len(g), batch_size)]

    dev = B.input_device(model)
    t0 = time.time()
    for bi, idx in enumerate(batches):
        ids = torch.tensor([ids_list[i] for i in idx], device=dev)
        am = torch.ones_like(ids)
        out = B.forward_logits(model, {"input_ids": ids, "attention_mask": am},
                               use_cache=True)
        raw = decode_digits(model, out.logits[:, -1, :].float(), out.past_key_values, am,
                            K * P.DIGITS_PER_LEVEL, digit_ids10)
        pmf_b, degen_b = digits_to_pmf(combine_digit_groups(raw, K), marg)
        pmf[idx], degenerate[idx], filled[idx] = pmf_b, degen_b, True
        copied[idx] = ["".join(map(str, r)) == example for r in raw]
        del out
        if bi % report_every == 0:
            done = int(filled.sum())
            print(f"  {done}/{n}  {done / max(time.time() - t0, 1e-9):.2f} items/s", flush=True)

    if not filled.all():
        raise RuntimeError(f"{key}: {int((~filled).sum())} items never filled")
    extra = {"n": n, "n_degenerate": int(degenerate.sum()),
             "degenerate_rate": float(degenerate.mean()),
             # the model reproduced the prompt's worked example verbatim: small
             # models do this a lot, and it is a capability limit, not a parse error
             "example_copy_rate": float(copied.mean()),
             "model": getattr(model.config, "_name_or_path", "unknown"),
             "batch_size": batch_size, "max_len": max_len,
             "grammar": "digits-fixed-count", "n_annotators": P.N_ANNOTATORS}
    save_result(path, key, ds, pmf, extra)
    el = time.time() - t0
    print(f"[{key}/{split}] {el / 60:.1f} min ({n / el:.2f} items/s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_data_args(ap)
    add_model_args(ap)
    ap.add_argument("--split", nargs="+", default=["val", "test"], choices=["val", "test"],
                    help="nothing is trained here, so train is never needed")
    ap.add_argument("--out-dir", default=None, help="default <run-root>/<model-slug>/")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip datasets already present in the split's JSON")
    a = ap.parse_args()

    out_dir = Path(a.out_dir) if a.out_dir else Path(a.run_root) / B.model_slug(a.model)
    print(f"model={a.model} batch={a.batch_size} splits={a.split} -> {out_dir}", flush=True)
    B.check_attention_indexing(a.model, a.batch_size, a.max_len, cache_dir=a.cache_dir)
    datasets = load_all(a)

    def done(split, key):
        p = out_dir / f"verbal_{split}.json"
        return p.exists() and key in json.load(open(p))

    todo = [(s, stem, ds) for stem, ds, _ in datasets for s in a.split
            if not (a.skip_existing and done(s, stem))]
    if not todo:
        print("nothing to do")
        return
    model, tok, _ = B.load_backbone(a.model, a.dtype, a.cache_dir)
    for split, stem, ds in todo:
        run_task(ds, stem, split, model, tok, a.batch_size, a.max_len,
                 out_dir / f"verbal_{split}.json")


if __name__ == "__main__":
    main()
