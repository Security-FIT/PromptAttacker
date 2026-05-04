import random
import re
from typing import Union, List, Dict, Any
from attacks.common.llm import LLM


class RALLM:

    def __init__(self, use_ollama=False, ollama_model=None, per_victim_llm=None):

        self.base_llm = LLM(model_path=per_victim_llm, ollama_model=ollama_model, use_ollama=False, temperature=0.8, max_tokens=512,)

        self.n = 3
        self.p = 0.2
        self.t = 0.6
        self.min_kept_tokens = 3
        self.refusal_patterns = [
            r"I'm sorry",
            r"I cannot",
            r"I am not able to",
            r"unfortunate",
            r"violate",
            r"security policy",
            r"as an AI"
        ]

    def _alignment_check(self, response: str) -> bool:
        """True = Fail (model refused the prompt)."""
        if not response:
            return False

        for pattern in self.refusal_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True

        return False

    def _perturb_prompt(self, prompt: str) -> str:
        """Randomly drop tokens while preserving order."""
        words = prompt.split()

        if len(words) < 5:
            return prompt

        kept = [w for w in words if random.random() > self.p]

        if len(kept) < self.min_kept_tokens:
            kept = words[:self.min_kept_tokens]

        return " ".join(kept)

    def _extract_user_prompt(self, messages: Union[str, List[Dict[str, Any]]]) -> str:
        if isinstance(messages, str):
            return messages

        return messages[-1]["content"]

    def response(self, messages: Union[str, List[Dict[str, Any]]]) -> str:

        original_prompt = self._extract_user_prompt(messages)

        # 1️⃣ test original prompt
        first_resp = self.base_llm.response(original_prompt)

        if self._alignment_check(first_resp):
            return "Blocked by RA-LLM"

        # 2️⃣ Monte Carlo perturbations
        perturbed_prompts = [
            self._perturb_prompt(original_prompt)
            for _ in range(self.n)
        ]

        responses = self.base_llm.response_batch(perturbed_prompts)

        fail_count = 0

        for r in responses:
            if self._alignment_check(r):
                fail_count += 1

        # 3️⃣ threshold decision
        if (fail_count / self.n) > self.t:
            return "Blocked by RA-LLM"

        # 4️⃣ allowed
        return self.base_llm.response(original_prompt)

    def response_batch(self, prompts: List[str]) -> List[str]:
        return [self.response(p) for p in prompts]