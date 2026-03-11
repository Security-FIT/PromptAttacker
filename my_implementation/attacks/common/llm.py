## @file llm.py
#  @brief Unified LLM wrapper for Ollama and vLLM backends
#
#  The goal of this wrapper is to provide a minimal, consistent interface
#  (`response()` / `response_batch()`) that can be reused across different
#  jailbreak attacks and evaluation runners without coupling the attack logic
#  to a specific inference backend.
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This file was fully designed and implemented by Bc. Petr Kaška.
#   - The interface design, backend switching logic, batching behavior,
#     and error-handling strategy are original work by the author.
#   - While the implementation builds on standard usage patterns of Ollama
#     and vLLM, no code was copied from existing attack repositories.


import requests
from vllm import LLM as VLLMClient, SamplingParams
from openai import OpenAI

class LLM:
    def __init__(self,
                 model_path: str,
                 temperature: float = 0.8,
                 max_tokens: int = 500,
                 ollama_model: str = "llama3.1:8b",
                 use_ollama: bool = True,
                 timeout: int = 1200
                 ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens > 1000 else 1000
        self.ollama_model = ollama_model
        self.use_ollama = use_ollama
        self.timeout = timeout

        if not self.use_ollama:
            self.client = VLLMClient(model=self.model_path,trust_remote_code=True)
        else:
            self.OLLAMA_HOST = "http://127.0.0.1:11434"

    def response(self, messages) -> str:
            if isinstance(messages, str):
                user_prompt = messages
            elif isinstance(messages, list) and messages and isinstance(messages[-1], dict) and "content" in messages[-1]:
                user_prompt = messages[-1]["content"]
            else:
                raise ValueError(f"Unsupported messages format: {type(messages)}")

            if self.use_ollama:
                payload = {
                    "model": self.ollama_model,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                }
                try:
                    resp = requests.post(
                        f"{self.OLLAMA_HOST}/api/generate",
                        json=payload,
                        timeout=self.timeout
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("response", "").strip()
                except requests.Timeout:
                    return f"[TIMEOUT ERROR] Model neodpověděl do {self.timeout} sekund."
                except requests.RequestException as e:
                    return f"[REQUEST ERROR] {e}"

            else:
                try:
                    outputs = self.client.generate(
                        [{"prompt": user_prompt}],
                        sampling_params=SamplingParams(
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                    )
                    return outputs[0].outputs[0].text.strip()
                except Exception as e:
                    return f"[ERROR] Chyba při vLLM inference: {e}"
                
    def response_batch(self, prompts: list[str]) -> list[str]:
        if self.use_ollama:
            return [self.response(p) for p in prompts]

        try:
            sampling = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            outputs = self.client.generate(
                prompts,
                sampling_params=sampling
            )

            results: list[str] = []
            for out in outputs:
                if not out.outputs:
                    results.append("[ERROR] vLLM nevrátil žádný výstup.")
                else:
                    results.append((out.outputs[0].text or "").strip())
            return results
        except Exception as e:
            err = f"[ERROR] Chyba při vLLM batch inference: {e}"
            return [err for _ in prompts]