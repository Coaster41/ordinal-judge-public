"""Build the quickstart sample from the raw HelpSteer2 `disagreements` file.

    python examples/make_sample.py --data-dir data              # both files
    python examples/make_sample.py --data-dir data --n 600

Writes, beside this script:
  hs2_helpfulness.jsonl     ~600 items, every annotator's helpfulness rating (M = 2..6)
  hs2_helpfulness_m1.jsonl  the same items with ONLY the first rating (M = 1)

Both use `hs2_helpfulness.spec.json`, whose template reproduces the paper's
HelpSteer2 prompt exactly (tests/test_data.py checks this when the raw file is
present). Whole prompts are sampled, so the two responses to a prompt stay in
the same split.

HelpSteer2 (Wang et al., 2024, NVIDIA) is released under CC-BY-4.0:
https://huggingface.co/datasets/nvidia/HelpSteer2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ojc import data as D

HERE = Path(__file__).resolve().parent
ATTR = "helpfulness"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data",
                    help="directory holding helpsteer2_disagreements.jsonl")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(Path(a.data_dir) / "helpsteer2_disagreements.jsonl")]
    recs = [{"id": f"hs2-{i}", "group_id": r["prompt"],
             "prompt": D._trunc(r["prompt"], 300), "response": D._trunc(r["response"], 400),
             "ratings": [int(s) for s in r[ATTR]]}
            for i, r in enumerate(rows) if len(r[ATTR]) >= 2]
    recs = D._subsample_by_group(recs, a.n, seed=a.seed)
    groups = {g: f"p{j}" for j, g in enumerate(dict.fromkeys(r["group_id"] for r in recs))}

    with open(HERE / "hs2_helpfulness.jsonl", "w") as f, \
         open(HERE / "hs2_helpfulness_m1.jsonl", "w") as f1:
        for r in recs:
            row = {"id": r["id"], "group": groups[r["group_id"]],
                   "prompt": r["prompt"], "response": r["response"]}
            f.write(json.dumps({**row, "ratings": r["ratings"]}) + "\n")
            f1.write(json.dumps({**row, "rating": r["ratings"][0]}) + "\n")
    print(f"wrote {len(recs)} items ({len(groups)} prompts) to {HERE}")


if __name__ == "__main__":
    main()
