"""Digit prompts: the logit rollout's and the verbal elicitation's.

Load-bearing details, each of which has cost a run:

* **Options are BARE digits 0..K-1.** " 0" (with a space) is two tokens in
  common tokenizers while "0" is one - the opposite of letters. The rollout
  reads a single logit position, so the prompt must end with a trailing space
  (`PREFILL = "Answer: "`) and the option token itself must be bare.
  `option_token_ids` asserts single-token and distinct for the loaded tokenizer.

* **Thinking is disabled.** `enable_thinking=False` is passed to the chat
  template; templates that do not use it ignore it. Without it, reasoning
  models open a `<think>` block and the answer position is somewhere else.

* **The prompt is never truncated from the right.** Truncation would cut the
  option list and the prefill, and the "answer" logit would be read from the
  middle of the item. `build_*_prompt` shrink the ITEM text until the whole
  prompt fits `max_len`.

* **No extra BOS.** Prompts are tokenised with `add_special_tokens=False`: the
  chat template already contains whatever BOS the model wants.
"""

from __future__ import annotations

import numpy as np
import torch

DIGITS = "0123456789"
PREFILL = "Answer: "          # trailing space is load-bearing; do not strip it
VERBAL_PREFILL = "Weights: "  # same reason

# Precision of the verbal histogram: "how many of N annotators chose each
# option", rendered as DIGITS_PER_LEVEL zero-padded digits per option. A single
# option taking all N renders as "99" - two digits cannot reach three.
N_ANNOTATORS = 100
DIGITS_PER_LEVEL = len(str(N_ANNOTATORS - 1))


def encode(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def option_token_ids(tok, K: int, device=None) -> torch.Tensor:
    """Token ids for the K option digits, asserted single-token and distinct."""
    ids = [tok.encode(DIGITS[i], add_special_tokens=False) for i in range(K)]
    bad = [(DIGITS[i], ids[i]) for i in range(K) if len(ids[i]) != 1]
    if bad:
        raise ValueError(f"option digits are not single-token for this tokenizer: {bad}")
    flat = [x[0] for x in ids]
    if len(set(flat)) != K:
        raise ValueError(f"option digit ids are not distinct: {flat}")
    return torch.tensor(flat, device=device)


def option_block(level_labels: list[str]) -> str:
    """The enumerated option list shown before the prefill, "  0 = worst"."""
    return "\n".join(f"  {DIGITS[i]} = {lbl}" if lbl else f"  {DIGITS[i]}"
                     for i, lbl in enumerate(level_labels))


def shorten(text: str, n_words: int) -> str:
    """Trim the MIDDLE of an item, keeping its head and the question at its tail."""
    w = text.split()
    if len(w) <= n_words:
        return text
    return f"{' '.join(w[: n_words - 40])} ... {' '.join(w[-40:])}"


def _chat(tok, body: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": body}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def _fit(tok, compose, item_text: str, max_len: int) -> str:
    s = compose(item_text)
    if len(encode(tok, s)) <= max_len:
        return s
    for words in (1200, 900, 600, 400, 250):
        s = compose(shorten(item_text, words))
        if len(encode(tok, s)) <= max_len:
            return s
    return compose(shorten(item_text, 150))


# --------------------------------------------------------------------------- #
# logit rollout
# --------------------------------------------------------------------------- #

def compose_logit(tok, item_text: str, level_labels: list[str]) -> str:
    body = f"{item_text}\n\nAnswer with a single digit.\n{option_block(level_labels)}"
    return _chat(tok, body) + PREFILL


def build_logit_prompt(tok, item_text: str, level_labels: list[str],
                       max_len: int = 2048) -> str:
    return _fit(tok, lambda t: compose_logit(tok, t, level_labels), item_text, max_len)


# --------------------------------------------------------------------------- #
# verbal elicitation
# --------------------------------------------------------------------------- #

def rescale_to_total(counts: np.ndarray, total: int = 10) -> np.ndarray:
    """Rescale a count vector to sum to exactly `total` by largest-remainder rounding."""
    c = np.asarray(counts, dtype=np.float64)
    if c.sum() <= 0:
        out = np.zeros(len(c), dtype=int)
        out[len(c) // 2] = total
        return out
    exact = c / c.sum() * total
    floor = np.floor(exact).astype(int)
    rem = total - floor.sum()
    if rem > 0:
        order = np.argsort(-(exact - floor))
        floor[order[:rem]] += 1
    return floor


def generic_example_digits(K: int) -> str:
    """A fixed, item-independent worked example: K counts out of N_ANNOTATORS,
    peaked near the middle, each as DIGITS_PER_LEVEL zero-padded digits, with no
    separator (position alone fixes each digit's place value). Exponential
    falloff, because a linear triangle rounds back to visually uniform."""
    center = (K - 1) / 2
    weights = np.array([2.0 ** (K - abs(i - center)) for i in range(K)])
    counts = rescale_to_total(weights, total=N_ANNOTATORS)
    return "".join(str(int(c)).zfill(DIGITS_PER_LEVEL) for c in counts)


def verbal_addendum(K: int) -> str:
    """Appended after the item text (which ends with a single-digit question):
    overrides that framing and supplies the worked example that keeps the
    digit-string format in-distribution before the grammar has to bind."""
    ex = generic_example_digits(K)
    total_digits = K * DIGITS_PER_LEVEL
    return (
        f"\n\nNow, instead of answering with a single digit, imagine {N_ANNOTATORS} "
        "independent annotators rate this independently. Report how many "
        f"(00-{N_ANNOTATORS - 1}) would choose each option, as exactly {K} "
        f"{DIGITS_PER_LEVEL}-digit zero-padded numbers back to back with no spaces or "
        f"punctuation ({total_digits} digits total). Example for a {K}-option scale: {ex}"
    )


def compose_verbal(tok, item_text: str, K: int) -> str:
    return _chat(tok, item_text + verbal_addendum(K)) + VERBAL_PREFILL


def build_verbal_prompt(tok, item_text: str, K: int, max_len: int = 2048) -> str:
    return _fit(tok, lambda t: compose_verbal(tok, t, K), item_text, max_len)
