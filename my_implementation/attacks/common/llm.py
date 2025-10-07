#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from vllm import LLM as VLLMClient, SamplingParams
from openai import OpenAI

class LLM:
    """
    Wrapper kolem vLLM nebo Ollama s podporou timeoutu.
    """
    def __init__(self,
                 model_path: str,
                 temperature: float = 0.8,
                 max_tokens: int = 512,
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
            self.client = VLLMClient(model=self.model_path)
        else:
            # Ollama poběží přes HTTP API
            self.OLLAMA_HOST = "http://127.0.0.1:11434"

    def response(self, messages: list[dict]) -> str:
        """Vrátí odpověď modelu s ošetřením timeoutu."""
        user_prompt = messages[-1]["content"]

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
            # vLLM lokální inference
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
