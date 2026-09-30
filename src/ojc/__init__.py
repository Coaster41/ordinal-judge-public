"""Ordinal LLM-as-a-judge calibration.

Pipeline (each step is a CLI, `python -m ojc.<step> --help`):

    extract      one forward pass per item -> activation + logit-rollout pmf
    verbal       verbalized annotator distribution (constrained digit decode)
    train_heads  climatology, ridge/spike, softmax and ordinal heads on the activations
    recalibrate  temperature scaling fitted on val, applied to test, for every arm
    evaluate     metrics for every arm (distributional, or point metrics at M = 1)
"""
