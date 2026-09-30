"""Integrity checks for the dataset loaders.

A silent data bug (a leaked group, a mis-mapped level, a dropped annotator)
would not raise anywhere downstream - it would just produce plausible wrong
numbers. `data.validate` is the generic check; run it on your own dataset too:

    python -c "from ojc import data as D; print(D.validate(D.load_jsonl('mine.jsonl')) or 'ok')"

The paper-corpus section runs only when the raw files are present (set
OJC_DATA_DIR, default ./data).

Run: python tests/test_data.py      (or: pytest tests/)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from ojc import data as D

PASS, FAIL, SKIP = [], [], []
ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def skip(name, why):
    SKIP.append(name)
    print(f"  [SKIP] {name}: {why}")


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def validate(ds: D.OrdinalDataset, min_m: int = 1, name: str | None = None) -> bool:
    bad = D.validate(ds, min_m)
    check(f"{name or ds.name}: passes data.validate", not bad, "; ".join(bad))
    return not bad


# --------------------------------------------------------------------------- #
print("\n1. The shipped sample")
ds = D.load_jsonl(EX / "hs2_helpfulness.jsonl")
validate(ds, min_m=2)
check("sample: 600 items, K=5, labels from the spec",
      len(ds) == 600 and ds.K == 5 and ds.level_labels == ["worst", "", "", "", "best"])
check("sample: every prompt ends with the 0..K-1 question",
      all(t.endswith("On a scale of 0 to 4, the helpfulness of this response is:")
          for t in ds.text))
ds1 = D.load_jsonl(EX / "hs2_helpfulness_m1.jsonl", EX / "hs2_helpfulness.spec.json")
validate(ds1, min_m=1)
check("M=1 sample: exactly one rating per item", (ds1.M == 1).all())
check("M=1 sample: same items, same splits, same text as the full sample",
      list(ds1.item_id) == list(ds.item_id) and list(ds1.split) == list(ds.split)
      and list(ds1.text) == list(ds.text))
check("M=1 sample: its rating is one of the full panel's",
      all(ds.counts[i, ds1.counts[i].argmax()] > 0 for i in range(len(ds))))
check("the M=1 file has no sidecar spec, and says so",
      _raises(lambda: D.load_jsonl(EX / "hs2_helpfulness_m1.jsonl")))

# --------------------------------------------------------------------------- #
print("\n2. load_jsonl: formats and validation")


def _write(tmp: Path, rows, spec, name="t"):
    p = tmp / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp / f"{name}.spec.json").write_text(json.dumps(spec))
    return p


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    rows = [{"id": f"q{i}", "question": f"question number {i}?", "answer": f"answer {i}",
             "ratings": [1 + i % 5, 1 + (i + 1) % 5], "group": f"g{i // 2}"} for i in range(40)]
    spec = {"K": 5, "levels": [1, 2, 3, 4, 5],
            "template": "Q: {question}\nA: {answer}\nOn a scale of 0 to 4, the answer is:"}
    d = D.load_jsonl(_write(td, rows, spec))
    validate(d, min_m=2, name="templated")
    check("raw levels 1..5 are mapped to 0..4",
          d.counts[0].tolist() == [1, 1, 0, 0, 0], f"{d.counts[0].tolist()}")
    check("the template is filled from each row's fields",
          d.text[3] == "Q: question number 3?\nA: answer 3\nOn a scale of 0 to 4, the answer is:")
    check("groups are respected by the automatic split",
          all(len({s for s, g in zip(d.split, d.group_id) if g == gg}) == 1
              for gg in set(d.group_id)))
    check("the split is deterministic",
          list(D.load_jsonl(td / "t.jsonl").split) == list(d.split))

    r2 = [{"text": f"item {i} is long enough", "rating": i % 3} for i in range(30)]
    d2 = D.load_jsonl(_write(td, r2, {"K": 3}, "single"))
    check("'rating' (M=1) and default ids by line number",
          (d2.M == 1).all() and d2.item_id[4] == "4")
    r3 = [{"text": f"item {i} is long enough", "counts": [i % 2, 1, 2]} for i in range(30)]
    d3 = D.load_jsonl(_write(td, r3, {"K": 3}, "counts"))
    check("'counts' is taken as given", d3.counts[1].tolist() == [1, 1, 2])
    r4 = [{"text": f"item {i} is long enough", "ratings": [0] * (1 + i % 3)} for i in range(30)]
    d4 = D.load_jsonl(_write(td, r4, {"K": 3, "min_annotators": 2}, "minm"))
    check("min_annotators drops thinner items", len(d4) == 20 and d4.M.min() >= 2)
    r5 = [{"text": f"item {i} is long enough", "rating": 0,
           "split": ["train", "val", "test"][i % 3]} for i in range(30)]
    d5 = D.load_jsonl(_write(td, r5, {"K": 2}, "given"))
    check("an explicit split column is used as given",
          list(d5.split[:3]) == ["train", "val", "test"])

    bad = [
        ("a rating outside the levels", [{"text": "xxxxxxxxxxxx", "rating": 7}], {"K": 3}),
        ("both ratings and rating", [{"text": "xxxxxxxxxxxx", "rating": 1, "ratings": [1]}], {"K": 3}),
        ("counts of the wrong length", [{"text": "xxxxxxxxxxxx", "counts": [1, 1]}], {"K": 3}),
        ("duplicate ids", [{"id": "a", "text": "xxxxxxxxxxxx", "rating": 1}] * 2, {"K": 3}),
        ("split on some lines only", [{"text": "xxxxxxxxxxxx", "rating": 1, "split": "train"},
                                      {"text": "xxxxxxxxxxxx", "rating": 1}], {"K": 3}),
        ("a group straddling splits", [{"text": "x" * 12, "rating": 1, "group": "g", "split": "train"},
                                       {"text": "x" * 12, "rating": 1, "group": "g", "split": "test"}],
         {"K": 3}),
        ("level labels of the wrong length", [{"text": "xxxxxxxxxxxx", "rating": 1}],
         {"K": 3, "level_labels": ["a", "b"]}),
        ("K above 10", [{"text": "xxxxxxxxxxxx", "rating": 1}], {"K": 11}),
    ]
    for i, (why, rr, sp) in enumerate(bad):
        check(f"rejects {why}", _raises(lambda: D.load_jsonl(_write(td, rr, sp, f"bad{i}"))))

leaky = D.load_jsonl(EX / "hs2_helpfulness.jsonl")
leaky.split[np.flatnonzero(leaky.group_id == leaky.group_id[0])[1:]] = (
    "test" if leaky.split[0] != "test" else "train")
check("data.validate catches a group straddling two splits",
      "no group straddles two splits" in D.validate(leaky))
check("resolve() routes .jsonl paths and task names",
      D.resolve(str(EX / "hs2_helpfulness.jsonl")).name == "hs2_helpfulness"
      and D.task_stem("hs2:helpfulness") == "hs2_helpfulness"
      and D.task_stem("x/y/mine.jsonl") == "mine")

# --------------------------------------------------------------------------- #
print("\n3. Paper corpora (raw files required)")
data_dir = Path(os.environ.get("OJC_DATA_DIR", "data"))
EXPECT_K = {"cb": 7, "hs2": 5, "mhs": 5, "d3code": 5}
for task in D.PAPER_TASKS:
    try:
        dsp = D.load(task, data_dir=data_dir)
    except (FileNotFoundError, ImportError) as e:
        skip(task, f"{type(e).__name__}: raw file not under {data_dir}")
        continue
    validate(dsp, min_m=2, name=task)
    check(f"{task}: K", dsp.K == EXPECT_K[task.split(":")[0]])
    check(f"{task}: reverse_rollout only for mhs:respect",
          dsp.reverse_rollout == (task == "mhs:respect"))
if (data_dir / "helpsteer2_disagreements.jsonl").exists():
    full = D.load("hs2:helpfulness", data_dir=data_dir)
    pos = {i: k for k, i in enumerate(full.item_id)}
    check("the shipped sample reproduces the paper's HelpSteer2 prompt and counts exactly",
          all(full.text[pos[i]] == t and (full.counts[pos[i]] == c).all()
              for i, t, c in zip(ds.item_id, ds.text, ds.counts)))


# --------------------------------------------------------------------------- #
def test_data():
    assert not FAIL, FAIL


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        sys.exit(1)
    print("dataset loaders verified")
