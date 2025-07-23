from __future__ import annotations

from typing import List, Dict, Any

from vllm import LLM as VLLMClient, SamplingParams


class LLM:
    """
    Lehký wrapper nad vLLM, který kromě jednoduché metody ``response`` zpřístupňuje
    i volání ``continual_generate`` a ``batch_generate`` – právě ty využívá TAP Attacker.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
        gpu_ids: list[int] | None = None,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        # vLLM dokáže automaticky zvolit CPU/GPU; pro více GPU se určuje tensor-parallel size
        self.client = VLLMClient(
            model=self.model_path,
            trust_remote_code=True,
        )

    # --------------------------------------------------------------------- #
    # public API – jednoduchý single-turn                                   #
    # --------------------------------------------------------------------- #
    def response(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Vrátí odpověď modelu na *poslední* uživatelskou zprávu."""
        prompt = messages[-1]["content"]
        outputs = self.client.generate(
            [{"prompt": prompt}],
            sampling_params=SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        return outputs[0].outputs[0].text