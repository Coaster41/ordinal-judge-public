"""Prompt construction against a real tokenizer.

Uses the tokenizer (CPU only) of OJC_TEST_MODEL, default Qwen/Qwen3.5-0.8B;
skips cleanly if it cannot be loaded (e.g. offline with an empty HF cache).

Run: python tests/test_prompts.py      (or: pytest tests/)
"""

from __future__ import annotations

import os
import sys

import numpy as np

from ojc import prompts as P

PASS, FAIL, SKIP = [], [], []
MODEL = os.environ.get("OJC_TEST_MODEL", "Qwen/Qwen3.5-0.8B")


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def skip(name, why):
    SKIP.append(name)
    print(f"  [SKIP] {name}: {why}")


# --------------------------------------------------------------------------- #
print("\n1. Tokenizer-free pieces")
for K in (2, 5, 7, 10):
    ex = P.generic_example_digits(K)
    c = np.array([int(ex[i:i + 2]) for i in range(0, len(ex), 2)])
    check(f"K={K}: worked example is K 2-digit groups summing to 100",
          len(ex) == 2 * K and c.sum() == P.N_ANNOTATORS and ex.isdigit(), ex)
for tot in (10, 100):
    r = P.rescale_to_total(np.array([3.0, 1.0, 1.0, 0.0]), total=tot)
    check(f"rescale_to_total sums to {tot} and keeps the order", r.sum() == tot
          and r[0] >= r[1] >= r[3], f"{r.tolist()}")
check("option_block renders anchors only where given",
      P.option_block(["worst", "", "best"]) == "  0 = worst\n  1\n  2 = best")
long = " ".join(f"w{i}" for i in range(3000))
sh = P.shorten(long, 400)
check("shorten keeps the head AND the tail (the question lives at the tail)",
      sh.startswith("w0 w1") and sh.endswith("w2999") and len(sh.split()) <= 401)

# --------------------------------------------------------------------------- #
print(f"\n2. With a real tokenizer ({MODEL})")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
except Exception as e:  # noqa: BLE001
    tok = None
    skip("tokenizer sections", f"{type(e).__name__}: {e}")

if tok is not None:
    ids = P.option_token_ids(tok, 10)
    check("digits 0-9 are single, distinct tokens", len(set(ids.tolist())) == 10)
    check("' 0' (with a space) is NOT the same token as '0' - why PREFILL ends in a space",
          tok.encode(" 0", add_special_tokens=False) != tok.encode("0", add_special_tokens=False))
    labels = ["worst", "", "", "", "best"]
    item = "Rate this.\n\nOn a scale of 0 to 4, the quality is:"
    s = P.build_logit_prompt(tok, item, labels)
    check("logit prompt ends with the prefill", s.endswith(P.PREFILL))
    check("logit prompt contains the item and the option list",
          item in s and P.option_block(labels) in s)
    if "<think>" in s:
        check("thinking is off: any <think> block is already CLOSED before the answer",
              s.rfind("</think>") > s.rfind("<think>"))
    else:
        check("thinking is off: no <think> block at all", True)
    v = P.build_verbal_prompt(tok, item, 5)
    check("verbal prompt ends with its prefill and carries the worked example",
          v.endswith(P.VERBAL_PREFILL) and P.generic_example_digits(5) in v)
    n_bos = 0
    if tok.bos_token_id is not None:
        n_bos = P.encode(tok, s).count(tok.bos_token_id)
    check("no doubled BOS when tokenising a templated prompt", n_bos <= 1, f"{n_bos}")
    huge = "word " * 20000 + "\n\nOn a scale of 0 to 4, the quality is:"
    sh = P.build_logit_prompt(tok, huge, labels, max_len=512)
    check("an over-long item is shortened to fit max_len, keeping question and prefill",
          len(P.encode(tok, sh)) <= 512 and sh.endswith(P.PREFILL)
          and "On a scale of 0 to 4, the quality is:" in sh,
          f"{len(P.encode(tok, sh))} tokens")


# --------------------------------------------------------------------------- #
def test_prompts():
    assert not FAIL, FAIL


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        sys.exit(1)
    print("prompt construction verified")
