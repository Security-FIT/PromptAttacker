import requests
from vllm import LLM as VLLMClient, SamplingParams
from openai import OpenAI

class LlamaGuardLLM:
    def __init__(self, ollama_model, guard_model="/storage/brno2/home/xkaska01/master/my_implementation/models/llamaguard3:8b"):
        self.base_llm = ollama_model
        self.guard_model = guard_model

    def _is_safe(self, prompt: str) -> tuple[bool, str]:
        # Využijeme stávající logiku LLM třídy, ale pro guard model
        payload = {
            "model": self.guard_model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            # Voláme přímo Ollama API pro Guard
            resp = requests.post(f"{self.base_llm.OLLAMA_HOST}/api/generate", json=payload, timeout=60)
            result = resp.json().get("response", "").strip()
            # Llama Guard vrací "unsafe\n{category}" nebo "safe"
            if "unsafe" in result.lower():
                return False, result
            return True, "safe"
        except Exception as e:
            return False, f"Guard Error: {e}"

    def response(self, messages) -> str:
        prompt = messages if isinstance(messages, str) else messages[-1]["content"]
        safe, verdict = self._is_safe(prompt)
        
        if not safe:
            return f"Blocked by Llama Guard: {verdict}"
        
        return self.base_llm.response(messages)