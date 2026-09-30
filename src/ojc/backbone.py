"""Loading a Hugging Face backbone and capturing its last-layer activation.

Works with any chat/instruct model that `transformers` can load for causal LM
(including multimodal wrappers whose text decoder sits at
`model.model.language_model`). Requirements, all checked at startup:

  * a chat template (the prompts go through `apply_chat_template`);
  * the digits 0..9 are single, distinct tokens WITHOUT a leading space
    (`prompts.option_token_ids` asserts this);
  * the forward accepts `logits_to_keep` (any recent transformers model does).

Load the backbone ONCE and reuse it across tasks: reloading per task leaves the
previous copy in the CUDA caching allocator and the third load OOMs.
"""

from __future__ import annotations

import torch

INT32_MAX = 2**31 - 1


def model_slug(model_name: str) -> str:
    """Directory name for a backbone's outputs: `Qwen/Qwen3.5-0.8B` -> `qwen3.5-0.8b`.

    Outputs are namespaced per backbone; without that a second model silently
    overwrites the first and the "comparison" is a model against itself.
    """
    return model_name.rstrip("/").split("/")[-1].lower()


def check_attention_indexing(model_name: str, batch_size: int, max_len: int,
                             limit_frac: float = 0.9, hard: bool = True,
                             cache_dir: str | None = None) -> float:
    """Reject a batch whose attention-scores tensor overflows int32 indexing.

    The scores tensor is (batch, heads, seq, seq). Once its element count reaches
    2**31 some attention kernels' index arithmetic wraps and reads out of bounds,
    which CUDA reports as an *illegal memory access*, not an OOM - it passes
    every memory check, survives a warm-up batch, and then faults on some later
    call. Worse, length-sorted batching puts the longest items in the same
    batch, so a clean run on a sample says nothing about the full corpus. The
    check is analytic, from `max_len`, with a 0.9 safety margin.
    """
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception as e:
        print(f"[attn] could not read config ({e}); skipping index check", flush=True)
        return 0.0
    cfg = getattr(cfg, "text_config", cfg)
    heads = getattr(cfg, "num_attention_heads", None)
    if not heads:
        return 0.0
    n = batch_size * heads * max_len * max_len
    frac = n / INT32_MAX
    safe = int(limit_frac * INT32_MAX) // (heads * max_len * max_len)
    print(f"[attn] {heads} heads, batch {batch_size}, seq {max_len} => "
          f"{frac:.2f} x INT32_MAX; largest safe batch {safe}", flush=True)
    if frac > limit_frac:
        msg = (f"--batch-size {batch_size} builds a {n:,}-element attention tensor for "
               f"{model_name} ({frac:.2f} x INT32_MAX), which faults as an illegal memory "
               f"access rather than an OOM. Largest safe batch at seq {max_len} is {safe}; "
               f"lower --batch-size or --max-len.")
        if hard:
            raise SystemExit(msg)
        print(f"[attn] WARNING: {msg}", flush=True)
    return frac


def get_text_model(model):
    """Resolve the text decoder across wrapper layouts (plain causal LM,
    multimodal `...ForConditionalGeneration`, GPT-2/NeoX-style `transformer`)."""
    for path in (("model", "language_model"), ("language_model",), ("model",),
                 ("transformer",), ("gpt_neox",)):
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "layers") or hasattr(obj, "h"):
            return obj
    if hasattr(model, "get_decoder"):
        return model.get_decoder()
    raise RuntimeError(f"could not locate the text decoder on {type(model).__name__}")


def final_norm_module(text_model):
    """The module whose output is `last_hidden_state`."""
    for name in ("norm", "final_layernorm", "ln_f", "final_layer_norm"):
        m = getattr(text_model, name, None)
        if isinstance(m, torch.nn.Module):
            return m, name
    raise RuntimeError(
        f"could not find the final norm on {type(text_model).__name__}; "
        f"candidates tried: norm, final_layernorm, ln_f, final_layer_norm"
    )


class FinalNormHook:
    """Capture the final norm's output at position -1, and nothing else.

    Registered on the text decoder's final norm, which is what produces
    `last_hidden_state`. Position -1 is the true final content token for every
    row because padding is LEFT. Capturing one slice rather than passing
    `output_hidden_states=True` avoids materialising every layer.

    `n_calls` is asserted by the caller: if a model ever routed through this
    module more than once per forward we would be storing the wrong pass.
    """

    def __init__(self, module: torch.nn.Module):
        self.h: torch.Tensor | None = None
        self.n_calls = 0
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, _mod, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        # detach + copy immediately: keeping a view would pin the whole activation
        self.h = t[:, -1, :].detach().float().cpu()
        self.n_calls += 1

    def reset(self):
        self.h, self.n_calls = None, 0

    def close(self):
        self._handle.remove()


def load_backbone(model_name: str, dtype: str = "bfloat16", cache_dir: str | None = None,
                  device_map: str = "auto"):
    """(model, tokenizer, text_decoder). `device_map="auto"` shards a model that
    does not fit on one GPU across all visible ones."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not getattr(tok, "chat_template", None):
        raise SystemExit(f"{model_name} has no chat template; use an instruct/chat model")

    kw = dict(dtype=getattr(torch, dtype), device_map=device_map, cache_dir=cache_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    except ValueError:
        # multimodal checkpoints (e.g. vision-language instruct models) register
        # under image-text-to-text rather than causal LM
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_name, **kw)
    model.eval()
    text_model = get_text_model(model)
    cfg = getattr(text_model, "config", model.config)
    cfg = getattr(cfg, "text_config", cfg)
    print(f"{model_name}: {type(model).__name__}, text decoder "
          f"{type(text_model).__name__} hidden={cfg.hidden_size}", flush=True)
    return model, tok, text_model


def hidden_size(model, text_model) -> int:
    cfg = getattr(text_model, "config", model.config)
    return getattr(cfg, "text_config", cfg).hidden_size


def forward_logits(model, enc, **kw):
    """Full-model forward returning only the last position's logits.

    Through the model's OWN head, never a hand-rolled `lm_head(h)`: some models
    (e.g. Gemma) soft-cap their logits in the head, and skipping it distorts the
    pmf. `logits_to_keep=1` keeps the logits tensor at (batch, 1, vocab) instead
    of (batch, seq, vocab).
    """
    kw.setdefault("use_cache", False)
    return model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                 logits_to_keep=1, **kw)


def input_device(model):
    """Where inputs go: the embedding's device under a sharded device_map."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return model.device
