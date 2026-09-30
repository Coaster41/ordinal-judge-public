#!/usr/bin/env bash
# The README quickstart as one script: a ~1B model on the 600-item sample.
#
#   bash examples/quickstart.sh                       # multi-rater sample (M = 2..6)
#   bash examples/quickstart.sh m1                    # single-rating sample (M = 1)
#   MODEL=meta-llama/Llama-3.2-1B-Instruct bash examples/quickstart.sh
#
# Needs one GPU (~4 GB for a 0.8B model in bf16). Outputs land in runs/<model-slug>/
# (runs/m1/<model-slug>/ for the M = 1 variant, so the two do not collide).
set -eo pipefail

MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}
BATCH=${BATCH:-16}
PY=${PY:-python}

if [[ "${1:-}" == "m1" ]]; then
  DATA=examples/hs2_helpfulness_m1.jsonl
  SPEC=(--spec examples/hs2_helpfulness.spec.json)   # the M = 1 file reuses the spec
  ROOT=runs/m1
else
  DATA=examples/hs2_helpfulness.jsonl
  SPEC=()
  ROOT=runs
fi
RUN=$ROOT/$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')

# 1. one forward pass per item: last-layer activation + logit-rollout pmf
$PY -m ojc.extract --model "$MODEL" --data "$DATA" "${SPEC[@]}" --batch-size "$BATCH" \
    --run-root "$ROOT"

# 2. verbalized distribution, on val (for recalibration) and test
$PY -m ojc.verbal --model "$MODEL" --data "$DATA" "${SPEC[@]}" --batch-size "$BATCH" \
    --run-root "$ROOT"

# 3. baselines + softmax/ordinal heads on the activations (CPU is fine at this size)
$PY -m ojc.train_heads --act-dir "$RUN/activations"

# 4. temperature scaling fitted on val, applied to test, for every arm
$PY -m ojc.recalibrate --run-dir "$RUN"

# 5. score everything: raw and recalibrated arms, including verbal
$PY -m ojc.evaluate --run-dir "$RUN" --preds-subdir preds_recal
