# Ordinal judge calibration

Calibrated LLM-as-a-judge scores for **ordinal** ratings (0..K-1), where each item's target
is the **distribution of its annotators' ratings**, not a single label.

The pipeline:

1. Run a frozen instruct LLM over each item **once**. That one forward pass gives two
   things:
   - the last-layer activation at the answer position;
   - the model's own zero-shot answer distribution over the digit tokens `0..K-1` (the
     *logit rollout*).
2. Separately, ask the model to *verbalize* how 100 annotators would split (the *verbal*
   arm).
3. Train small heads on the frozen activations:
   - a **softmax** head (any pmf);
   - an **ordinal** cumulative-link head (respects the ordering).

   Each is fit with an L2 ridge whose strength is tuned on validation, next to three
   baselines: climatology, ridge→spike, and the rollout.
4. **Temperature-scale** every arm on validation.
5. **Evaluate** everything with distributional scoring rules. When there is only one
   rating per item, point metrics are used instead.

Scope: this is the core of the research code. The last-layer Laplace posterior and its
aleatoric/epistemic split are not included yet.

```
src/ojc/
  data.py         datasets: generic JSONL loader + the paper's four corpora
  backbone.py     HF model loading, final-norm hook, int32 attention guard
  prompts.py      digit prompts (logit rollout and verbal)
  extract.py      CLI  stage 1: activations + logit-rollout pmf
  verbal.py       CLI  verbalized annotator distribution (constrained digit decode)
  heads.py        softmax and ordinal (cumulative-link) heads
  losses.py       M-weighted soft CE, RPS, Brier, threshold-BCE
  ridge.py        standardiser, closed-form weighted ridge path, spike, climatology
  train_heads.py  CLI  stage 2: every baseline and head, per-item predictions saved
  recalibrate.py  CLI  temperature scaling (pmf/cdf families) fit on val, applied to test
  metrics.py      debiased RPS, ECE_top / ECE_thr + null floor, point metrics, ...
  evaluate.py     CLI  score every arm of a run
examples/         600-item HelpSteer2 sample (+ M=1 variant), spec, quickstart.sh
tests/            6 suites; CPU parts run anywhere, GPU parts skip without a GPU
```

## Install

You need Python ≥ 3.10 (tested on 3.10) and one CUDA GPU for the LLM steps. Heads
train fine on CPU.

```bash
git clone <this repo> ordinal-judge && cd ordinal-judge
python -m venv .venv && source .venv/bin/activate

# 1. torch first, from the index that matches your CUDA driver (see pytorch.org), e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2. this package; [paper] adds pandas/pyarrow for the paper corpora, [dev] adds pytest
pip install -e ".[paper,dev]"
# optional: the exact versions the quickstart numbers below were produced with
pip install -r requirements.txt
```

Two environment gotchas:

- **Python dev headers.** Triton, which torch and some model kernels use, JIT-compiles a
  small CUDA module on first GPU use and needs `Python.h`.
  - If the first forward dies with a compiler error, install your distro's `python3-dev`,
    or use a Python that ships its headers (conda does).
  - Check with:
    `python -c "import sysconfig,os;print(os.path.exists(sysconfig.get_paths()['include']+'/Python.h'))"`.
- **Compiler under conda.** If a conda env is on `PATH`, Triton may pick up conda's
  compiler and fail to link. Run `export CC=/usr/bin/gcc` first.

Models download through the normal Hugging Face cache. Set `HF_HUB_CACHE` to put it
somewhere else, or pass `--cache-dir`.

## Quickstart: a ~1B model on the sample

Model: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B). It is ungated,
about 1.7 GB, and needs about 4 GB of GPU memory in bf16.

Data: `examples/hs2_helpfulness.jsonl`, 600 HelpSteer2 responses, each rated for
helpfulness on 0–4 by 2–6 annotators.

```bash
bash examples/quickstart.sh          # all five steps below, about 5 min on one RTX A5000
```

or step by step:

```bash
M=Qwen/Qwen3.5-0.8B
D=examples/hs2_helpfulness.jsonl

# 1. one forward pass per item -> runs/qwen3.5-0.8b/activations/hs2_helpfulness.pt
python -m ojc.extract --model $M --data $D --batch-size 16

# 2. verbal elicitation on val and test -> runs/qwen3.5-0.8b/verbal_{val,test}.json
python -m ojc.verbal --model $M --data $D --batch-size 16

# 3. baselines + heads -> runs/qwen3.5-0.8b/{results.json, preds/}
python -m ojc.train_heads --act-dir runs/qwen3.5-0.8b/activations

# 4. temperature scaling, fit on val -> runs/qwen3.5-0.8b/preds_recal/ (+ recal_params.json)
python -m ojc.recalibrate --run-dir runs/qwen3.5-0.8b

# 5. metrics for every arm, raw and "+ts" -> runs/qwen3.5-0.8b/eval_recal.{md,json}
python -m ojc.evaluate --run-dir runs/qwen3.5-0.8b --preds-subdir preds_recal
```

Step 5 prints a table like this (test split, 120 items; abridged):

| arm | RPS(deb) | MAE | ECE_thr | ECE_thr-floor | ECE_top | CE | acc |
|---|---|---|---|---|---|---|---|
| climatology | 0.3445 ± 0.0515 | 0.9728 ± 0.0598 | 0.0252 | 0.0084 | 0.0000 | 1.4789 | 0.4583 |
| logit_rollout | 0.3752 ± 0.0567 | 0.9259 ± 0.0701 | 0.1286 | 0.0901 | 0.1350 | 1.6217 | 0.3750 |
| spike_tuned | 0.4117 ± 0.0416 | 1.0442 ± 0.0492 | 0.0917 | 0.0733 | 0.3100 | 1.8123 | 0.1583 |
| softmax_map__ce | 0.3077 ± 0.0436 | 0.9338 ± 0.0544 | 0.0383 | 0.0034 | 0.0357 | 1.4313 | 0.4583 |
| ordinal_map__rps | 0.2959 ± 0.0415 | 0.9111 ± 0.0529 | 0.0666 | 0.0228 | 0.0843 | 1.4354 | 0.4667 |
| verbal | 1.0961 ± 0.1112 | 1.6664 ± 0.1054 | 0.3510 | 0.3383 | 0.4521 | 9.2693 | 0.1583 |
| logit_rollout+ts | 0.3256 ± 0.0466 | 0.9419 ± 0.0565 | 0.0662 | 0.0286 | 0.0413 | 1.4941 | 0.3750 |
| softmax_map__ce+ts | 0.3120 ± 0.0465 | 0.9176 ± 0.0578 | 0.0666 | 0.0318 | 0.0469 | 1.4349 | 0.4583 |
| verbal+ts | 0.4717 ± 0.0417 | 1.2308 ± 0.0562 | 0.1719 | 0.1503 | 0.0371 | 1.6094 | 0.1583 |

How to read it:

- **Lower is better**, except `acc`.
- `RPS(deb)` is the headline scoring rule.
- `ECE_thr-floor` near 0 means no detectable miscalibration.
- A `+ts` row is the same arm after temperature scaling.

Two caveats for this sample:

- **120 test items is a smoke test, not an experiment.** The bootstrap SEs on RPS are
  shown for that reason.
- **A 0.8B model is a weak judge.** Its zero-shot rollout and verbal arms are poor. The
  point of the quickstart is to exercise the pipeline end to end.
  - The heads trained on its activations score better than climatology (0.296–0.312
    vs 0.345). In a paired bootstrap over test items the gap is 0.03–0.05, and its
    95% interval excludes zero for three of the four heads, but not for
    `ordinal_map__rps`.
  - Its verbal answers are mostly the prompt's worked example copied verbatim:
    `verbal.py` logs this as `example_copy_rate`, 64% here.
  - Temperature scaling then flattens the verbal arm almost to uniform. A larger model is
    needed for that arm to mean anything.

Every CLI has `--help`. `--data` takes several datasets at once, and the model is loaded
only once.

## Bring your own data

One JSON object per line, plus a spec file named `<name>.spec.json` next to it:

```jsonl
{"id": "q1", "question": "What is 2+2?", "answer": "4, obviously.", "ratings": [4, 4, 3], "group": "math-1"}
{"id": "q2", "question": "What is 2+2?", "answer": "Probably 5.",   "ratings": [0, 1],    "group": "math-1"}
```

```json
{
  "K": 5,
  "levels": [0, 1, 2, 3, 4],
  "level_labels": ["useless", "", "", "", "perfect"],
  "template": "Rate the answer.\n\nQuestion: {question}\n\nAnswer: {answer}\n\nOn a scale of 0 to 4, the quality of this answer is:"
}
```

What each field means:

- **`ratings`** holds one entry per annotator. Instead of `ratings`, a line can give
  `"rating": 3` (a single label) or `"counts": [0, 0, 1, 1, 2]` (per-level counts).
- **`levels`** lists your raw rating values in order (for example `[1,2,3,4,5]`, or
  `["bad","ok","good"]`). They are mapped to 0..K-1. It defaults to `0..K-1`.
- **`template`** is filled with `str.format` from each line's fields. Without a template,
  each line needs a `"text"` field, which is used as is.
  - **End the text with the question on the 0..K-1 scale.** The prompt builder appends
    `Answer with a single digit.`, the digit options with your `level_labels`, and the
    answer prefill `Answer: `.
  - K can be 2–10.
- **`group`** (optional) keeps related items, such as several answers to one question, in
  the same split.
- **`split`** (optional, but all-or-none) fixes the split yourself. Without it, groups are
  split 60/20/20 with a fixed seed (`split_seed` and `split_fracs` in the spec).
- **Other spec keys:**
  - `min_annotators` (default 1) drops thinner items;
  - `reverse_rollout: true` is for a prompt whose scale points the opposite way to your
    stored levels (see `train_heads.align_rollout`).

Sanity-check a new file with
`python -c "from ojc import data as D; print(D.validate(D.load_jsonl('mine.jsonl')) or 'ok')"`. Outputs are named after the file stem.

**The paper's corpora** load by name instead of by path: `--data cb hs2:helpfulness
mhs:respect d3code ...` (all names are in `ojc.data.PAPER_TASKS`). They need the raw
files under `--data-dir` (default `./data`); `src/ojc/data.py` lists which files.

### Single-label data (M = 1)

This works everywhere, with no flags. When some item has only one rating, `evaluate`
switches to the **point** regime and says so in the table header:

- **Undefined:** `RPS(deb)`, which needs M ≥ 2 to remove the finite-panel bias. With one
  label you cannot see annotator disagreement, only a model's calibration against single
  draws of it.
- **Headline:** accuracy (pmf mode vs label), MAE of the pmf median, quadratic-weighted
  kappa, Spearman, NLL of the observed label, raw RPS (= discrete CRPS against the label),
  and top-label ECE.
- **Still valid:** `ECE_thr` and its null floor.
- **Unchanged:** training and temperature scaling. At M = 1, M-weighted cross entropy is
  ordinary cross entropy, and the recalibrator is classic temperature scaling.
- **Mixed M:** if some items have M ≥ 2, `RPS(deb)` is also reported on that subset.

```bash
bash examples/quickstart.sh m1   # same items, one rating each -> runs/m1/qwen3.5-0.8b/
```

## Bring your own model

`--model` takes any Hugging Face id or local path that loads as a causal LM. It also
accepts multimodal instruct models, whose text decoder is found automatically. The loader
checks three requirements at startup:

- it has a **chat template**;
- the digits `0`–`9` are **single, distinct tokens without a leading space** (verified for
  Qwen3.5/3.6 and Gemma-4; the check raises before anything runs if a tokenizer fails it);
- the forward accepts **`logits_to_keep`** (any recent `transformers` model).

On multiple GPUs, `device_map="auto"` shards a large model across all visible cards.

If `transformers` logs that it is "falling back to its reference PyTorch implementation"
(Qwen3.5's linear-attention layers do this), the results are correct but slower.
`pip install flash-linear-attention causal-conv1d` enables the fast kernels. The quickstart
numbers above were produced without them.
Logits always come from the model's own head, so logit soft-capping (Gemma) is applied.

`--batch-size` guidance:

- **Keep it fixed across any runs you compare.** bf16 forward passes are not
  batch-invariant: an item's activation moves by a small fraction of a feature SD with
  batch composition.
- **`--max-len`** (default 2048) is the prompt token budget. Long items are shortened
  in the middle to fit. If a prompt still does not fit, `extract` stops rather than
  cut off the answer prefill; raise `--max-len`.
- **Big models at long `--max-len`:** a startup check rejects batch sizes whose attention
  tensor would overflow int32 indexing. That overflow shows up as an *illegal memory
  access*, not an OOM, several batches in.

## Outputs

`runs/<model-slug>/` (the slug is the model name's last path part, lowercased):

| file | contents |
|---|---|
| `activations/<task>.pt` | `X` (n×d float32 activations), `logit_pmf` (n×K), `counts` (n×K), `split`, `item_id`, `group_id`, `K`, `level_labels`, `reverse_rollout`, `model`, source info for `extract.verify` |
| `verbal_{val,test}.json` | `{task: {item_id, pmf (n×K), degenerate_rate, ...}}` |
| `results.json` | one row per (task, arm): val + test metrics, selected λ, fit diagnostics |
| `preds/<task>.pt` | `pred_val` / `pred_test`: `{arm: n×K pmf}`, plus `counts_*` and `item_id_*`. Any metric can be recomputed, or any arm recalibrated, without refitting |
| `preds_recal/<task>.pt` | the same with a `<arm>+ts` twin per arm (verbal included); `recal_params.json` has each fitted β and temperature |
| `eval*.{md,json}` | the evaluation table and every metric |

Arms:

- `climatology` / `climatology_itemavg`: the training marginal, no features;
- `logit_rollout`: zero-shot, nothing fitted;
- `spike_0.9` / `spike_tuned`: ridge → panel mean, α of the mass on the nearest level;
- `softmax_map__{ce,rps}` and `ordinal_map__{ce,rps}`: the heads, fit under M-weighted
  soft CE or RPS (`--losses` also accepts `brier`, `tbce`);
- `verbal`.

## Metrics, briefly

- **RPS(deb).** The ranked probability score, Σₖ (F̂ₖ − Gₖ)². Its finite-panel bias
  Σ G(1−G)/(M−1) is removed per item, so the number estimates the RPS against the
  *population* rating distribution, and items with different M are comparable. It needs
  M ≥ 2.
- **MAE.** |mean of the pmf − mean rating|.
- **ECE_thr.** Binned calibration of the K−1 cumulative events {y ≤ k}, which is
  ordinal-aware. Binned ECE is biased upward even for a perfect model, so it is reported
  with its **null floor**: the ECE of a perfectly calibrated model on the same items and
  panel sizes. Read `ECE_thr-floor`.
- **ECE_top.** Top-label calibration; at M = 1 it is the classic ECE.
- **Temperature.** The CLI's default is `pmf` family, 1 parameter, CE objective:
  p ← softmax(β·log p), fit on val.
  - `recal_params.json` reports β **and** `temperature = 1/β`. Temperature > 1 means the
    arm was overconfident.
  - Beware `recalibrate.params()["T"]`: it is β, not the temperature.
  - `--family cdf --n-params 2 --objective rps` gives the ordinal-aware alternative.

## Tests

```bash
pytest tests/                 # or run any file directly for per-check output
python tests/test_extract.py  # GPU sections use OJC_TEST_MODEL (default Qwen/Qwen3.5-0.8B)
```

What each suite covers:

| suite | covers |
|---|---|
| `test_math` | the scoring rules' bias claims, ECE and its null floor, loss properness, M = 1 point metrics |
| `test_recalibrate` | the recalibration families |
| `test_heads` | ridge, λ selection, and end-to-end `run_task` → recalibrate → evaluate on toy files, including M = 1 |
| `test_data` | the loaders and `data.validate` |
| `test_prompts` | prompt construction against the real tokenizer |
| `test_extract` | the length-sorted batching **permutation test** and the hook ≡ `last_hidden_state` check on the GPU |

The permutation test matters most: a wrong inverse permutation would silently attach every
activation to the wrong item.

## Data licence

The sample is derived from HelpSteer2 (Wang et al., 2024, NVIDIA), licensed CC-BY-4.0:
https://huggingface.co/datasets/nvidia/HelpSteer2. `examples/make_sample.py` rebuilds it
from the raw `disagreements` file.
