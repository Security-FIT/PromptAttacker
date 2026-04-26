from typing import Union, List, Dict, Any
from attacks.common.llm import LLM


class LlamaGuard:
    def __init__(
        self,
        use_ollama=False,
        ollama_model=None,
        per_victim_llm=None,
        guard_model=None,
        temperature=0.8,
        max_tokens=512,
    ):
        # self.base_llm = LLM(
        #     model_path=per_victim_llm,
        #     ollama_model=ollama_model,
        #     use_ollama=False,
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        # )
        # # guard model
        self.guard_llm = LLM(
            model_path="/storage/brno2/home/xkaska01/master/my_implementation/models/llamaguard3:8b",
            ollama_model=ollama_model,
            use_ollama=False,
            temperature=0.0,
            max_tokens=64,
        )

    def _extract_user_prompt(self, messages: Union[str, List[Dict[str, Any]]]) -> str:
        if isinstance(messages, str):
            return messages

        if not messages:
            return ""

        return messages[-1]["content"]

    def _is_safe(self, prompt: str):
        try:
            result = self.guard_llm.response(prompt).strip().lower()

            if "unsafe" in result:
                return False, result
            if "safe" in result:
                return True, result

            return False, f"Unexpected guard response: {result}"

        except Exception as e:
            return False, f"Guard Error: {e}"

    def response(self, messages: Union[str, List[Dict[str, Any]]]) -> str:
        prompt = self._extract_user_prompt(messages)

        safe, verdict = self._is_safe(prompt)
        if not safe:
            return f"Blocked by Llama Guard: {verdict}"

        return str(safe) + " -  " + verdict

    def response_batch(self, prompts: List[str]) -> List[str]:
        return [self.response(prompt) for prompt in prompts]