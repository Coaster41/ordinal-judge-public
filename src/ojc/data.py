"""Datasets for ordinal-annotation targets.

Every dataset is reduced to one schema, so everything downstream is K-agnostic:

    item_id       str            unique item key
    group_id      str            grouping key for leak-free splits
    text          str            the item text handed to the backbone (the chat
                                 template, option list and answer prefill are
                                 added later by `ojc.prompts`)
    counts        (n, K) int     per-level annotator counts
    M             (n,) int       annotators for this item (= counts.sum(1))
    split         str            train / val / test
    level_labels  list[str]      K short anchors shown next to the digits
                                 ("worst", "", "", "", "best"); "" = no anchor
    reverse_rollout bool         the prompt's scale points the opposite way to
                                 the counts (see train_heads.align_rollout)

`counts` (not a normalised pmf) is the primitive: the multinomial log-likelihood
is defined on counts, and the M_i weighting used by the losses falls straight
out of it. M = 1 (one rating per item) is allowed; see `metrics.regime`.

Two ways in:

  * `load_jsonl(path)` - YOUR data. One JSON object per line plus a sidecar
    `<name>.spec.json`; format documented on the function and in the README.
  * `load(task, data_dir)` - the four corpora from the paper (`cb`,
    `hs2:<attr>`, `mhs:<item>`, `d3code`), read from their raw files.

`resolve(source)` accepts either and is what every CLI uses.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEED = 0
SPLIT_FRACS = (0.6, 0.2, 0.2)
SPLITS = ("train", "val", "test")


@dataclass
class OrdinalDataset:
    name: str
    K: int
    item_id: np.ndarray
    group_id: np.ndarray
    text: np.ndarray
    counts: np.ndarray  # (n, K) int
    split: np.ndarray
    level_labels: list = field(default_factory=list)
    reverse_rollout: bool = False

    def __post_init__(self):
        if not self.level_labels:
            self.level_labels = [""] * self.K
        if len(self.level_labels) != self.K:
            raise ValueError(f"{self.name}: {len(self.level_labels)} level labels for K={self.K}")
        if not 2 <= self.K <= 10:
            raise ValueError(f"{self.name}: K={self.K}; the digit prompts support 2..10 levels")

    @property
    def M(self) -> np.ndarray:
        return self.counts.sum(1)

    @property
    def phat(self) -> np.ndarray:
        """Empirical pmf. Never use this for the loss - use counts."""
        return self.counts / self.counts.sum(1, keepdims=True)

    def subset(self, split: str) -> "OrdinalDataset":
        m = self.split == split
        return OrdinalDataset(
            self.name, self.K, self.item_id[m], self.group_id[m],
            self.text[m], self.counts[m], self.split[m],
            list(self.level_labels), self.reverse_rollout,
        )

    def __len__(self) -> int:
        return len(self.item_id)


def _group_split(group_id: np.ndarray, seed: int = SEED,
                 fracs: tuple = SPLIT_FRACS) -> np.ndarray:
    """Assign train/val/test by *group*, so no group straddles two splits.

    Many corpora re-use a source across items (several responses to one prompt,
    several summaries of one article); a row-level split would leak. Items with
    no natural grouping use their own id as the group.
    """
    groups = np.unique(group_id)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(groups))
    n_tr = int(round(fracs[0] * len(groups)))
    n_va = int(round(fracs[1] * len(groups)))
    assign = {}
    for rank, gi in enumerate(order):
        assign[groups[gi]] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
    return np.array([assign[g] for g in group_id])


def _subsample_by_group(recs: list, n_max: int | None, seed: int = SEED) -> list:
    """Take whole GROUPS at random until n_max items are reached.

    Sampling groups rather than rows keeps the split machinery honest (no group
    is half-sampled) and, being uniform over groups, leaves the M distribution
    unbiased. Note that because `_group_split` permutes the groups PRESENT,
    changing n_max changes every item's split.
    """
    if n_max is None or len(recs) <= n_max:
        return recs
    groups = sorted({r["group_id"] for r in recs})
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    by_group: dict = {}
    for r in recs:
        by_group.setdefault(r["group_id"], []).append(r)
    out = []
    for g in groups:
        if len(out) >= n_max:
            break
        out.extend(by_group[g])
    return out[:n_max]


def _trunc(s: str, n_words: int) -> str:
    return " ".join(s.split()[:n_words])


def validate(ds: OrdinalDataset, min_m: int = 1) -> list[str]:
    """The integrity checks every dataset must pass; returns the failures (empty = ok).

    A silent data bug - a leaked group, a mis-mapped level, a dropped annotator -
    raises nowhere downstream; it just produces plausible wrong numbers.
    """
    g = {s: set(ds.group_id[ds.split == s]) for s in SPLITS}
    checks = {
        "counts are non-negative and sum to M": (ds.counts >= 0).all()
        and (ds.counts.sum(1) == ds.M).all(),
        f"every item has >= {min_m} rating(s)": ds.M.min() >= min_m,
        "K matches the count matrix and the level labels":
            ds.counts.shape[1] == ds.K == len(ds.level_labels),
        "splits partition the items": sum((ds.split == s).sum() for s in SPLITS) == len(ds),
        "no group straddles two splits": not ((g["train"] & g["val"]) or (g["train"] & g["test"])
                                              or (g["val"] & g["test"])),
        "every split is non-empty": all(len(v) > 0 for v in g.values()),
        "text is present": all(isinstance(t, str) and len(t) > 10 for t in ds.text),
        "item ids are unique": len(set(ds.item_id)) == len(ds),
    }
    return [k for k, v in checks.items() if not v]


# --------------------------------------------------------------------------- #
# generic JSONL
# --------------------------------------------------------------------------- #

def spec_path_for(path: Path) -> Path:
    """`data/foo.jsonl` -> `data/foo.spec.json`."""
    return path.with_suffix(".spec.json")


def load_jsonl(path: str | Path, spec: str | Path | dict | None = None) -> OrdinalDataset:
    """Your own dataset: one JSON object per line, plus a spec.

    Each line:
        {"id": "q17",                      # optional; default = line number
         "text": "...",                    # the item, OR any fields the spec's
                                           #   "template" refers to
         "ratings": [3, 4, 4, 2],          # one entry per annotator; OR
         "rating": 3,                      #   a single rating (M = 1); OR
         "counts": [0, 0, 1, 1, 2],        #   per-level counts, length K
         "group": "prompt-5",              # optional; items sharing a group
                                           #   never straddle two splits
         "split": "train"}                 # optional; all-or-none

    The spec (`<name>.spec.json` beside the JSONL unless given explicitly):
        {"K": 5,                                    # number of ordinal levels, 2..10
         "levels": [1, 2, 3, 4, 5],                 # optional: raw rating values in
                                                    #   order; default 0..K-1
         "level_labels": ["worst", "", "", "", "best"],   # optional anchors
         "template": "Rate this answer.\\n\\nQuestion: {question}\\n\\n"
                     "Answer: {answer}\\n\\nOn a scale of 0 to 4, the quality is:",
                                                    # optional; str.format over
                                                    #   each line; default {text}
         "reverse_rollout": false,                  # optional; see data docstring
         "min_annotators": 1,                       # optional; drop items with
                                                    #   fewer ratings
         "split_seed": 0, "split_fracs": [0.6, 0.2, 0.2]}   # optional

    The item text should END with the question, phrased on the 0..K-1 scale
    ("On a scale of 0 to 4, ... is:"): the prompt builder appends the digit
    options and the answer prefill after it. Ratings are always mapped to
    levels 0..K-1, whatever their raw values.
    """
    path = Path(path)
    if spec is None:
        spec = spec_path_for(path)
    if not isinstance(spec, dict):
        sp = Path(spec)
        if not sp.exists():
            raise FileNotFoundError(f"no spec for {path}: expected {sp} (see load_jsonl)")
        spec = json.load(open(sp))
    K = int(spec["K"])
    raw_levels = spec.get("levels", list(range(K)))
    if len(raw_levels) != K:
        raise ValueError(f"spec 'levels' has {len(raw_levels)} entries for K={K}")
    to_level = {v: i for i, v in enumerate(raw_levels)}
    template = spec.get("template")
    min_m = int(spec.get("min_annotators", 1))

    ids, groups, texts, counts, splits = [], [], [], [], []
    n_dropped = 0
    for ln, line in enumerate(open(path)):
        if not line.strip():
            continue
        r = json.loads(line)
        where = f"{path.name}:{ln + 1}"
        given = [k for k in ("ratings", "rating", "counts") if k in r]
        if len(given) != 1:
            raise ValueError(f"{where}: need exactly one of ratings/rating/counts, got {given}")
        if "counts" in r:
            c = np.asarray(r["counts"], dtype=np.int64)
            if c.shape != (K,) or (c < 0).any():
                raise ValueError(f"{where}: counts must be {K} non-negative ints")
        else:
            vals = r["ratings"] if "ratings" in r else [r["rating"]]
            c = np.zeros(K, dtype=np.int64)
            for v in vals:
                if v not in to_level:
                    raise ValueError(f"{where}: rating {v!r} not in levels {raw_levels}")
                c[to_level[v]] += 1
        if c.sum() < max(min_m, 1):
            n_dropped += 1
            continue
        iid = str(r.get("id", ln))
        ids.append(iid)
        groups.append(str(r.get("group", iid)))
        texts.append(template.format_map(r) if template else r["text"])
        counts.append(c)
        splits.append(r.get("split"))

    if not ids:
        raise ValueError(f"{path}: no usable items")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: item ids are not unique")
    if n_dropped:
        print(f"[{path.stem}] dropped {n_dropped} items with fewer than "
              f"{max(min_m, 1)} ratings", flush=True)
    group_id = np.array(groups)
    have = [s is not None for s in splits]
    if all(have):
        bad = sorted({s for s in splits} - set(SPLITS))
        if bad:
            raise ValueError(f"{path}: unknown split values {bad}")
        split = np.array(splits)
        leak = {g for g in set(groups)
                if len({s for s, gg in zip(splits, groups) if gg == g}) > 1}
        if leak:
            raise ValueError(f"{path}: {len(leak)} groups straddle two splits, e.g. {sorted(leak)[:3]}")
    elif any(have):
        raise ValueError(f"{path}: 'split' is given on some lines but not others")
    else:
        split = _group_split(group_id, seed=int(spec.get("split_seed", SEED)),
                             fracs=tuple(spec.get("split_fracs", SPLIT_FRACS)))
    return OrdinalDataset(
        name=path.stem, K=K, item_id=np.array(ids), group_id=group_id,
        text=np.array(texts), counts=np.stack(counts), split=split,
        level_labels=list(spec.get("level_labels", [""] * K)),
        reverse_rollout=bool(spec.get("reverse_rollout", False)),
    )


# --------------------------------------------------------------------------- #
# the paper's corpora
# --------------------------------------------------------------------------- #
#
# Raw files, expected under `data_dir` (default ./data):
#   cb_all.csv                         CommitmentBank, all annotations
#   helpsteer2_disagreements.jsonl     HelpSteer2 `disagreements/` per-rater lists
#   measuring_hate_speech.parquet      Measuring Hate Speech (ucberkeley-dlab)
#   d3code_{items,ratings}.csv         D3Code (google-research-datasets/D3code)
#
# All four end their item text with a 0..K-1 scale sentence, so the answer is a
# single bare digit whatever the corpus's native coding.

DEFAULT_DATA_DIR = Path("data")

CB_K = 7  # answers -3..+3


def load_commitmentbank(data_dir: Path = DEFAULT_DATA_DIR) -> OrdinalDataset:
    """CommitmentBank-All.csv - one row per (item, annotator).

    `Answer` is on a 7-point -3..+3 scale, mapped to levels 0..6. Annotator
    count varies per item (8..18), which is exactly why the losses are
    M-weighted. Split is by item (`uID`) and stratified by genre.
    """
    rows = list(csv.DictReader(open(Path(data_dir) / "cb_all.csv")))

    per_item: dict[str, dict] = {}
    for r in rows:
        ans = r["Answer"].strip()
        if not (ans.lstrip("-").isdigit()):
            continue  # a handful of blank responses
        a = int(ans)
        if not -3 <= a <= 3:
            raise ValueError(f"answer {a} outside -3..3")
        uid = r["uID"]
        if uid not in per_item:
            per_item[uid] = {
                "counts": np.zeros(CB_K, dtype=np.int64),
                "genre": r["genre"],
                "text": _cb_prompt(r["Context"], r["Target"], r["Prompt"]),
            }
        per_item[uid]["counts"][a + 3] += 1

    uids = sorted(per_item)
    counts = np.stack([per_item[u]["counts"] for u in uids])
    genre = np.array([per_item[u]["genre"] for u in uids])
    text = np.array([per_item[u]["text"] for u in uids])

    # Stratify by genre: split within each genre, then concatenate, so every
    # split sees BNC/SWBD/WSJ in the same proportion.
    split = np.empty(len(uids), dtype=object)
    for g in np.unique(genre):
        m = genre == g
        split[m] = _group_split(np.array(uids)[m])

    return OrdinalDataset(
        name="commitmentbank", K=CB_K,
        item_id=np.array(uids), group_id=np.array(uids), text=text,
        counts=counts, split=split.astype(str),
        level_labels=["speaker certain it is false", "", "", "unknown", "", "",
                      "speaker certain it is true"],
    )


def _cb_prompt(context: str, target: str, prompt: str) -> str:
    return (
        "Read the passage and judge how strongly the speaker is committed to the "
        "embedded proposition being true.\n\n"
        f"Passage:\n{context} {target}\n\n"
        f"Proposition: {prompt}\n\n"
        # Levels are stated as 0..K-1 rather than the corpus's native -3..+3 so the
        # answer is a single token: "-3" costs several tokens, and the logit rollout
        # reads a single position.
        "On a scale from 0 (the speaker is certain it is false) through 3 (unknown) "
        "to 6 (the speaker is certain it is true), the speaker's commitment is:"
    )


# ---- HelpSteer2 ------------------------------------------------------------ #

HS2_ATTRS = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
HS2_K = 5  # scores 0..4

# Attribute definitions as given to the HelpSteer2 annotators (Wang et al., 2024).
HS2_RUBRIC = {
    "helpfulness": ("Helpfulness - overall helpfulness of the response to the prompt."),
    "correctness": ("Correctness - inclusion of all pertinent facts without errors."),
    "coherence": ("Coherence - consistency and clarity of expression."),
    "complexity": ("Complexity - intellectual depth required to write the response, from "
                   "language a child could write to requiring deep domain expertise."),
    "verbosity": ("Verbosity - amount of detail included in the response, relative to what "
                  "is asked for in the prompt."),
}


def load_helpsteer2(attr: str, n_max: int | None = None,
                    data_dir: Path = DEFAULT_DATA_DIR) -> OrdinalDataset:
    """HelpSteer2 `disagreements` split: per-annotator score lists, M varies 2-6.

    The main HelpSteer2 release stores one aggregated int per attribute and is
    therefore unusable here; the `disagreements/` file keeps the raw per-rater
    lists. Grouped by prompt (2 responses share each prompt).
    """
    if attr not in HS2_ATTRS:
        raise ValueError(f"unknown attribute {attr!r}, expected one of {HS2_ATTRS}")
    rows = [json.loads(line) for line in open(Path(data_dir) / "helpsteer2_disagreements.jsonl")]

    recs = []
    for i, r in enumerate(rows):
        scores = r[attr]
        if len(scores) < 2:
            continue
        c = np.zeros(HS2_K, dtype=np.int64)
        for s in scores:
            s = int(s)
            if not 0 <= s <= HS2_K - 1:
                raise ValueError(f"score {s} outside 0..{HS2_K-1}")
            c[s] += 1
        recs.append({
            "item_id": f"hs2-{i}", "group_id": r["prompt"], "counts": c,
            "text": hs2_prompt(r["prompt"], r["response"], attr),
        })

    recs = _subsample_by_group(recs, n_max, seed=SEED)
    group_id = np.array([r["group_id"] for r in recs])
    return OrdinalDataset(
        name=f"helpsteer2-{attr}", K=HS2_K,
        item_id=np.array([r["item_id"] for r in recs]), group_id=group_id,
        text=np.array([r["text"] for r in recs]),
        counts=np.stack([r["counts"] for r in recs]), split=_group_split(group_id),
        level_labels=["worst", "", "", "", "best"],
    )


def hs2_prompt(prompt: str, response: str, attr: str) -> str:
    """Item content first, rubric and question last."""
    return (
        "You are rating the quality of an AI assistant's response.\n\n"
        f"User prompt:\n{_trunc(prompt, 300)}\n\n"
        f"Assistant response:\n{_trunc(response, 400)}\n\n"
        f"{HS2_RUBRIC[attr]}\n\n"
        f"On a scale of 0 to 4, the {attr} of this response is:"
    )


# ---- Measuring Hate Speech ------------------------------------------------- #

MHS_ITEMS = ("respect", "dehumanize", "violence")
MHS_K = 5

# Agree/disagree survey items; the published codebook wording is reproduced so
# the prompt matches what annotators saw.
MHS_STATEMENT = {
    "respect": "This comment is respectful toward the group or person it refers to.",
    "dehumanize": "This comment dehumanizes the group or person it refers to "
                  "(compares them to something non-human, e.g. an animal or disease).",
    "violence": "This comment calls for or endorses violence against the group or person "
                "it refers to.",
}

# The raw MHS columns are all SEVERITY scales (4 = worst), but the `respect`
# statement is phrased positively, so for that item alone the digit 4 means
# "respectful" to the model and "disrespectful" in the counts. Trained arms
# absorb this; the zero-shot rollout does not (corr(rollout mean, human mean)
# = -0.77 uncorrected). The counts are left as the raw column and the model's
# pmf is reversed at scoring time instead (`train_heads.align_rollout`).
MHS_REVERSED = ("respect",)


def load_mhs(item: str, n_max: int | None = None,
             data_dir: Path = DEFAULT_DATA_DIR) -> OrdinalDataset:
    """Measuring Hate Speech: one row per (comment, annotator) Likert response.

    Grouped by comment_id. Panels are capped at 20 (a handful of calibration
    comments were rated by hundreds and would dominate the M-weighted loss).
    """
    if item not in MHS_ITEMS:
        raise ValueError(f"unknown item {item!r}, expected one of {MHS_ITEMS}")
    import pandas as pd
    df = pd.read_parquet(Path(data_dir) / "measuring_hate_speech.parquet")
    df = df[["comment_id", "text", item]].dropna()

    recs = []
    for cid, g in df.groupby("comment_id", sort=True):
        vals = g[item].astype(int).values
        if len(vals) < 2:
            continue
        if len(vals) > 20:
            vals = vals[:20]
        c = np.zeros(MHS_K, dtype=np.int64)
        for v in vals:
            if not 0 <= v <= MHS_K - 1:
                raise ValueError(f"level {v} outside 0..{MHS_K-1}")
            c[v] += 1
        recs.append({
            "item_id": str(cid), "group_id": str(cid), "counts": c,
            "text": _mhs_prompt(g["text"].iloc[0], item),
        })

    recs = _subsample_by_group(recs, n_max, seed=SEED)
    group_id = np.array([r["group_id"] for r in recs])
    return OrdinalDataset(
        name=f"mhs-{item}", K=MHS_K,
        item_id=np.array([r["item_id"] for r in recs]), group_id=group_id,
        text=np.array([r["text"] for r in recs]),
        counts=np.stack([r["counts"] for r in recs]), split=_group_split(group_id),
        level_labels=["strongly disagree", "", "neutral", "", "strongly agree"],
        reverse_rollout=item in MHS_REVERSED,
    )


def _mhs_prompt(text: str, item: str) -> str:
    return (
        "Read the comment and rate your agreement with the statement below.\n\n"
        f"Comment:\n{_trunc(text, 220)}\n\n"
        f"Statement: {MHS_STATEMENT[item]}\n\n"
        "On a scale from 0 (strongly disagree) through 2 (neutral) "
        "to 4 (strongly agree), your response is:"
    )


# ---- D3Code ---------------------------------------------------------------- #

D3CODE_K = 5  # 0 (not offensive at all) .. 4 (extremely offensive)

D3CODE_STATEMENT = (
    "How offensive is this text to the group or person it refers to, on a scale "
    "from 0 (not at all offensive) to 4 (extremely offensive)?"
)


def load_d3code(n_max: int | None = None, data_dir: Path = DEFAULT_DATA_DIR) -> OrdinalDataset:
    """D3Code (Davani et al. 2024): parallel offensiveness ratings of 4,554 items
    by 4k+ raters across 21 countries.

    `rating_raw == -1` (rater did not understand the item) is dropped. Panel size
    is wide (9-196) by design and is not capped. `items.csv` has duplicate
    item_id rows with identical text; the first occurrence is kept. Split
    stratified by `category`.
    """
    data_dir = Path(data_dir)
    items, categories = {}, {}
    for r in csv.DictReader(open(data_dir / "d3code_items.csv")):
        if r["item_id"] in items:
            continue
        items[r["item_id"]] = r["text"]
        categories[r["item_id"]] = r["category"]

    counts: dict[str, np.ndarray] = {}
    for r in csv.DictReader(open(data_dir / "d3code_ratings.csv")):
        raw = int(r["rating_raw"])
        if raw == -1:
            continue
        if not 0 <= raw <= D3CODE_K - 1:
            raise ValueError(f"rating {raw} outside 0..{D3CODE_K-1}")
        iid = r["item_id"]
        if iid not in items:
            continue
        if iid not in counts:
            counts[iid] = np.zeros(D3CODE_K, dtype=np.int64)
        counts[iid][raw] += 1

    recs = [
        {"item_id": iid, "group_id": iid, "counts": c, "text": _d3code_prompt(items[iid]),
         "category": categories[iid]}
        for iid, c in counts.items() if c.sum() >= 2
    ]

    recs = _subsample_by_group(recs, n_max, seed=SEED)
    item_id = np.array([r["item_id"] for r in recs])
    category = np.array([r["category"] for r in recs])

    split = np.empty(len(recs), dtype=object)
    for cat in np.unique(category):
        m = category == cat
        split[m] = _group_split(item_id[m])

    return OrdinalDataset(
        name="d3code", K=D3CODE_K,
        item_id=item_id, group_id=item_id,
        text=np.array([r["text"] for r in recs]),
        counts=np.stack([r["counts"] for r in recs]), split=split.astype(str),
        level_labels=["not offensive at all", "", "", "", "extremely offensive"],
    )


def _d3code_prompt(text: str) -> str:
    return (
        "Read the text below and judge how offensive it is.\n\n"
        f"Text:\n{_trunc(text, 220)}\n\n"
        f"{D3CODE_STATEMENT}"
    )


PAPER_TASKS = (["cb"]
               + [f"hs2:{a}" for a in HS2_ATTRS]
               + [f"mhs:{i}" for i in MHS_ITEMS]
               + ["d3code"])


def load(task: str, data_dir: Path = DEFAULT_DATA_DIR, n_max: int | None = None) -> OrdinalDataset:
    """One of the paper's corpora by task name (see PAPER_TASKS)."""
    if task.startswith("hs2:"):
        return load_helpsteer2(task.split(":", 1)[1], n_max=n_max, data_dir=data_dir)
    if task.startswith("mhs:"):
        return load_mhs(task.split(":", 1)[1], n_max=n_max, data_dir=data_dir)
    if task in ("cb", "commitmentbank"):
        return load_commitmentbank(data_dir=data_dir)
    if task == "d3code":
        return load_d3code(n_max=n_max, data_dir=data_dir)
    raise ValueError(f"unknown task {task!r}; expected a .jsonl path or one of {PAPER_TASKS}")


# --------------------------------------------------------------------------- #
# one entry point for the CLIs
# --------------------------------------------------------------------------- #

def is_jsonl(source: str) -> bool:
    return str(source).endswith(".jsonl")


def task_stem(source: str) -> str:
    """The colon-free name every output file for this dataset is called by."""
    if is_jsonl(source):
        return Path(source).stem
    return source.replace(":", "_")


def resolve(source: str, spec: str | None = None, data_dir: Path = DEFAULT_DATA_DIR,
            n_max: int | None = None) -> OrdinalDataset:
    """A `.jsonl` path (with its spec) or a paper task name -> OrdinalDataset."""
    if is_jsonl(source):
        return load_jsonl(source, spec)
    return load(source, data_dir=data_dir, n_max=n_max)
