"""Guards for the single-pass activation + logit-rollout path.

The dangerous failure here is silent: length-sorted batching processes items out
of order, so a wrong inverse permutation would attach every activation to the
wrong item and produce a file that passes `extract.verify` (which compares the
item_id LIST, never that rows correspond to items). Section 2c is that check.

Section 1 is CPU-only. Section 2 needs a GPU and OJC_TEST_MODEL (default
Qwen/Qwen3.5-0.8B); it skips cleanly, and says so, when either is missing.

Run: python tests/test_extract.py      (or: pytest tests/)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from ojc import backbone as B
from ojc import data as D
from ojc import extract as X
from ojc import prompts as P
from ojc import verbal as V

PASS, FAIL, SKIP = [], [], []
MODEL = os.environ.get("OJC_TEST_MODEL", "Qwen/Qwen3.5-0.8B")
EX = Path(__file__).resolve().parent.parent / "examples"


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def skip(name, why):
    SKIP.append(name)
    print(f"  [SKIP] {name}: {why}")


# --------------------------------------------------------------------------- #
print("\n1. length-sorted batching partitions exactly and is order-independent")
rng = np.random.RandomState(0)
for n, bs in [(1, 8), (7, 8), (8, 8), (100, 8), (1000, 16), (23, 5)]:
    L = rng.randint(5, 2000, size=n)
    batches = X.sorted_batches(L, bs)
    flat = np.concatenate(batches) if batches else np.array([], dtype=int)
    check(f"n={n} bs={bs}: every index exactly once",
          sorted(flat.tolist()) == list(range(n)), f"{len(batches)} batches")
    check(f"n={n} bs={bs}: no batch exceeds bs", all(len(b) <= bs for b in batches))
    check(f"n={n} bs={bs}: batches are length-homogeneous",
          all((np.diff(L[b]) >= 0).all() for b in batches))
L = rng.randint(5, 2000, size=257)
payload = rng.randn(257, 4)
got = np.zeros_like(payload)
for b in X.sorted_batches(L, 16):
    got[b] = payload[b]          # exactly what run_task does: X[idx] = ...
check("scatter-by-index round trip is the identity", np.array_equal(got, payload))
check("verbal digit groups decode by place value",
      V.combine_digit_groups(np.array([[0, 5, 1, 2, 9, 9]]), 3).tolist() == [[5, 12, 99]])
pm, deg = V.digits_to_pmf(np.array([[0, 0, 0], [1, 1, 2]]), np.array([0.2, 0.3, 0.5]))
check("an all-zero verbal row falls back to the marginal and is flagged",
      deg.tolist() == [True, False] and np.allclose(pm[0], [0.2, 0.3, 0.5])
      and np.allclose(pm[1], [0.25, 0.25, 0.5]))

# --------------------------------------------------------------------------- #
print(f"\n2. GPU: {MODEL}")
if not torch.cuda.is_available():
    skip("GPU sections", "no CUDA device")
else:
    try:
        model, tok, text = B.load_backbone(MODEL)
    except Exception as e:  # noqa: BLE001
        model = None
        skip("GPU sections", f"could not load {MODEL}: {type(e).__name__}: {e}")

    if model is not None:
        dev = B.input_device(model)
        norm, norm_name = B.final_norm_module(text)
        hook = B.FinalNormHook(norm)
        # 24 REAL, distinct items of varied length: synthetic near-duplicates
        # ("word word ...") have final states so alike (cos ~0.9998) that bf16
        # batch noise can reorder them and the test would measure noise
        sample = D.load_jsonl(EX / "hs2_helpfulness.jsonl")
        order = np.argsort([len(t) for t in sample.text])
        pick = order[np.linspace(0, len(order) - 1, 24).astype(int)]
        texts = [P.build_logit_prompt(tok, sample.text[i], sample.level_labels, 1024)
                 for i in pick]

        # --- 2a: the hook IS last_hidden_state, and logits_to_keep is a slice ----
        enc = tok(texts[:4], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        hook.reset()
        with torch.no_grad():
            out = B.forward_logits(model, enc)
        check("hook fires exactly once per forward", hook.n_calls == 1, f"{hook.n_calls}")
        hooked = hook.h.clone()
        with torch.no_grad():
            ref = text(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                       use_cache=False).last_hidden_state[:, -1].float().cpu()
            full = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         use_cache=False).logits[:, -1].float()
        check(f"hooked activation ({norm_name}) == last_hidden_state[:, -1]",
              torch.allclose(hooked, ref, atol=1e-3, rtol=1e-3),
              f"maxdiff={float((hooked - ref).abs().max()):.3e}")
        check("logits_to_keep=1 yields one position",
              out.logits.shape[1] == 1, f"{tuple(out.logits.shape)}")
        check("...equal to the last position of the full logits",
              torch.allclose(out.logits[:, -1].float(), full, atol=1e-2, rtol=1e-3),
              f"maxdiff={float((out.logits[:, -1].float() - full).abs().max()):.3e}")
        oid = P.option_token_ids(tok, 5).to(full.device)
        pmf = torch.softmax(full[:, oid], -1)
        check("digit pmf is a valid distribution", torch.allclose(pmf.sum(-1), torch.ones(4, device=pmf.device)))

        # --- 2c: THE permutation test --------------------------------------------
        lens = np.array([len(P.encode(tok, t)) for t in texts])
        H = np.zeros((len(texts), B.hidden_size(model, text)), dtype=np.float32)
        for idx in X.sorted_batches(lens, 5):
            e = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                    add_special_tokens=False).to(dev)
            hook.reset()
            with torch.no_grad():
                B.forward_logits(model, e)
            H[idx] = hook.h.numpy()
        probe = [0, 7, 13, 23]
        ok, detail = True, []
        for i in probe:
            e = tok([texts[i]], return_tensors="pt", add_special_tokens=False).to(dev)
            hook.reset()
            with torch.no_grad():
                B.forward_logits(model, e)
            s = hook.h.numpy()[0]
            cos = [float(s @ H[k] / (np.linalg.norm(s) * np.linalg.norm(H[k]) + 1e-12))
                   for k in range(len(texts))]
            ok &= int(np.argmax(cos)) == i
            detail.append(f"item{i}: self={cos[i]:.4f} best_other="
                          f"{max(c for k, c in enumerate(cos) if k != i):.4f}")
        check("each row matches ITS OWN item best (inverse permutation is correct)", ok,
              "; ".join(detail))

        # --- 2d: run_task end to end on a slice of the sample, then verify --------
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rows = [json.loads(line) for line in open(EX / "hs2_helpfulness.jsonl")][:30]
            for i, r in enumerate(rows):
                r["split"] = ["train", "val", "test"][i % 3]
                r.pop("group")
            (td / "mini.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            (td / "mini.spec.json").write_text((EX / "hs2_helpfulness.spec.json").read_text())
            ds = D.load_jsonl(td / "mini.jsonl")
            meta = X._source_meta(str(td / "mini.jsonl"), None, D.DEFAULT_DATA_DIR, None)
            path = X.run_task(ds, "mini", meta, model, tok, text, hook, 7, 2048, td / "act")
            blob = torch.load(path, weights_only=False)
            check("run_task writes a file that verifies against its source", X.verify(path))
            check("...with the schema train_heads reads",
                  {"X", "logit_pmf", "counts", "split", "item_id", "K", "task", "model",
                   "reverse_rollout"} <= set(blob) and blob["X"].shape[0] == 30)
            # one item recomputed alone must match its stored row
            e = tok([P.build_logit_prompt(tok, ds.text[11], ds.level_labels)],
                    return_tensors="pt", add_special_tokens=False).to(dev)
            hook.reset()
            with torch.no_grad():
                o = B.forward_logits(model, e)
            single = hook.h.numpy()[0]
            stored = blob["X"].numpy()
            cos = stored @ single / (np.linalg.norm(stored, axis=1) * np.linalg.norm(single))
            check("the stored row for item 11 is the one closest to item 11 run alone",
                  int(np.argmax(cos)) == 11, f"self={cos[11]:.4f}")
            p1 = torch.softmax(o.logits[0, -1].float()[P.option_token_ids(tok, 5).to(o.logits.device)], -1)
            check("...and its stored logit pmf matches within bf16 batch noise",
                  float((blob["logit_pmf"][11] - p1.double().cpu()).abs().max()) < 0.05,
                  f"maxdiff={float((blob['logit_pmf'][11] - p1.double().cpu()).abs().max()):.3f}")

            # --- 2e: verbal decode on the same slice ------------------------------
            V.run_task(ds, "mini", "test", model, tok, 4, 2048, td / "verbal_test.json")
            vj = json.load(open(td / "verbal_test.json"))["mini"]
            vp = np.array(vj["pmf"])
            check("verbal writes one valid pmf per test item",
                  vp.shape == (10, 5) and np.allclose(vp.sum(1), 1.0)
                  and vj["item_id"] == list(ds.subset("test").item_id))
            check("verbal degenerate rate is reported", 0.0 <= vj["degenerate_rate"] <= 1.0,
                  f"{vj['degenerate_rate']:.2f}")
        hook.close()


# --------------------------------------------------------------------------- #
def test_extract():
    assert not FAIL, FAIL


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if SKIP:
        for s in SKIP:
            print("  skipped:", s)
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        sys.exit(1)
    print("activation-path guards passed")
