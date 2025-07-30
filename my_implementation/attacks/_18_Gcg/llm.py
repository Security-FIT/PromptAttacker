"""LLM wrapper _using 🤗 Transformers_ so that gradient‑based attacks
(GCG, EOT, …) can access model weights.

✅  Provides required attributes:
    • ``self.model``  – `transformers.PreTrainedModel` (with `.backward()`)
    • ``self.tokenizer``  – `transformers.PreTrainedTokenizer`

🔧  Designed as a drop‑in replacement for the previous vLLM wrapper. Signature
     stays compatible with the call in *main.py* (extra args are ignored).

Example
-------
>>> victim = LLM("./models/Llama-2-7b-hf", temperature=0.7, max_tokens=512)
>>> msg = [{"role": "user", "content": "Napiš haiku o Brně"}]
>>> print(victim.response(msg))
"Brněnská katedrála…"
"""

from __future__ import annotations

from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["LLM"]


class LLM:
    """Simple HF‑based local inference wrapper.

    Extra positional/keyword arguments are silently ignored so that existing
    code like ``LLM(path, temp, max_tokens, what_ollama_model, api_url)`` keeps
    working even though they are unused in this implementation.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
        *_,  # swallow unused positional args
        device: str | None = None,
        load_in_8bit: bool | None = None,
        **__,  # swallow unused kwargs
    ) -> None:
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        # ──────────────────────────────────────────────
        #  Tokenizer (always on CPU, fast=True if available)
        # ──────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
        )

        # ──────────────────────────────────────────────
        #  Model – choose device & dtype automatically
        # ──────────────────────────────────────────────
        #  * device_map="auto" lets HF/accelerate decide where to place layers
        #  * torch_dtype="auto" picks fp16/bf16 if GPU supports it
        #  * optional 8‑bit loading for VRAM‑poor setups
        # ──────────────────────────────────────────────
        model_kwargs: Dict[str, Any] = dict(
            trust_remote_code=True,
            device_map="auto" if device is None else {"": device},
            torch_dtype="auto",
        )
        if load_in_8bit is True:
            model_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()

        # convenient shortcut used by some other components (e.g. GCG)
        self.device = next(self.model.parameters()).device

    # ──────────────────────────────────────────────
    #  Chat‑style response (single‑user turn)
    # ──────────────────────────────────────────────
    def response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a completion for the **last** user message."""

        if not messages or messages[-1]["role"] != "user":
            raise ValueError("`messages` must end with a user message")

        prompt = messages[-1]["content"]
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **input_ids,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
