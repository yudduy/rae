"""Transformers-backed Actor that can return residual-stream activations.

Mirrors the `ActorClient.chat(messages, ...)` API so existing `compound_program`
code can use it transparently. Adds `chat_and_capture(...)` that generates a
response AND returns hidden-state vectors at (a) the final prompt token (the
Actor's state after reading the question) and (b) the final generated token
(the Actor's state right after producing the draft).

We use the last-token, all-layers residual stream because that's the cheapest
object rich enough to linearly probe `advisability = E[J(advise) - J(preserve) | h]`.

Design constraints:
- Deterministic (temperature=0) for reproducible probes.
- Single forward pass for activation capture (no re-running the model).
- Works with HF cache at /home/ubuntu/capr-cache/hf/ (set via
  `HF_HOME` env var or the ActivationActorConfig.cache_dir field).
- Chat template applied via tokenizer.apply_chat_template to match vLLM's
  chat behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ActivationActorConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 2048
    temperature: float = 0.0
    # Layers to persist. None = all layers; a list picks specific layer indices.
    capture_layers: Optional[list[int]] = None
    cache_dir: Optional[str] = None  # honors HF_HOME if None


@dataclass
class HiddenState:
    """Residual stream at one position, stacked across layers.

    Shape: (n_layers, hidden_size).  float32 on CPU to keep training cheap.
    """

    prompt_last: "object"  # numpy array after save
    gen_last: "object"


class ActivationActor:
    """Qwen-style transformers wrapper with an `ActorClient`-compatible `.chat()`.

    Not thread-safe (single GPU, single forward queue).
    """

    def __init__(self, cfg: Optional[ActivationActorConfig] = None):
        self.cfg = cfg or ActivationActorConfig()
        # Lazy imports so unit tests that mock this class don't need torch.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        tok_kwargs = {}
        mdl_kwargs: dict = {
            "torch_dtype": getattr(torch, self.cfg.dtype),
            "device_map": self.cfg.device,
        }
        if self.cfg.cache_dir:
            tok_kwargs["cache_dir"] = self.cfg.cache_dir
            mdl_kwargs["cache_dir"] = self.cfg.cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, **tok_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id, **mdl_kwargs
        )
        self.model.eval()
        self._pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    # ---- low-level helpers ---------------------------------------------------

    def _render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _gen(self, prompt_text: str, *, max_new_tokens: Optional[int] = None) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=max(self.cfg.temperature, 1e-5),
                pad_token_id=self._pad_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ---- ActorClient-compatible chat -----------------------------------------

    def chat(self, messages: list[dict], **kwargs) -> str:
        prompt_text = self._render(messages)
        return self._gen(prompt_text, max_new_tokens=kwargs.get("max_tokens"))

    # ---- Activation capture --------------------------------------------------

    def chat_and_capture(self, messages: list[dict], **kwargs):
        """Generate a response AND capture hidden states.

        Returns (response_text, HiddenState) where HiddenState has two numpy
        arrays shape (n_layers, hidden_size) for the final prompt token and
        the final generated token.
        """
        import numpy as np

        torch = self._torch
        prompt_text = self._render(messages)
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").to(
            self.model.device
        )
        prompt_len = prompt_ids["input_ids"].shape[1]

        with torch.inference_mode():
            gen_out = self.model.generate(
                **prompt_ids,
                max_new_tokens=kwargs.get("max_tokens", self.cfg.max_new_tokens),
                do_sample=self.cfg.temperature > 0,
                temperature=max(self.cfg.temperature, 1e-5),
                pad_token_id=self._pad_id,
                return_dict_in_generate=True,
            )

        full_ids = gen_out.sequences  # (1, prompt_len + new_len)
        new_ids = full_ids[0, prompt_len:]
        response_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # Second forward for clean hidden states over [prompt || draft].
        with torch.inference_mode():
            fwd = self.model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                output_hidden_states=True,
                use_cache=False,
            )
        # fwd.hidden_states: tuple length (n_layers + 1); each (1, seq_len, hidden)
        hidden = fwd.hidden_states
        n_layers = len(hidden)

        prompt_last_idx = prompt_len - 1
        gen_last_idx = full_ids.shape[1] - 1

        layers = self.cfg.capture_layers or list(range(n_layers))

        def _stack(idx: int):
            # (n_selected_layers, hidden_size), float32 cpu numpy
            vecs = [
                hidden[i][0, idx, :].to(torch.float32).cpu().numpy() for i in layers
            ]
            return np.stack(vecs, axis=0)

        h = HiddenState(
            prompt_last=_stack(prompt_last_idx), gen_last=_stack(gen_last_idx)
        )
        return response_text, h
