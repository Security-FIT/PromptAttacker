# TENTO FILE JE MUUUUUUUUUUJ

from vllm import LLM as VLLMClient, SamplingParams
from openai import OpenAI

class LLM:
    """
    Very simple wrapper around vLLM for local inference.

    Example:
        victim = LLM(
            model_path="./models/Llama-2-13b-chat",  # lokální složka s config.json apod.
            temperature=0.8,
            max_tokens=512,
            gpu_ids=[0],  # nebo [] pro CPU-only
        )
        response = victim.response([{"role":"user","content":"Ahoj, jak se máš?"}])
    """
    def __init__(self,
                 model_path: str,
                 temperature: float = 0.8,
                 max_tokens: int = 512,
                 ollama_model = "llama3.1:8b",
                 use_ollama = True
                 ):
        # cesta k lokálnímu HF modelu
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_model = ollama_model
        self.use_ollama = use_ollama


        if self.use_ollama:
            # nakonfigurujeme OpenAI klient na lokální Ollamu
            self.client = OpenAI(
                api_key="ollama",
                base_url="http://127.0.0.1:11434/v1"
            )
        else:
            # klasický vLLM klient
            self.client = VLLMClient(model=self.model_path)

        # vytvoříme klienta až v response(), abychom mohli znovu použít wrapper

    def response(self, messages: list[dict]) -> str:
        # očekáváme seznam zpráv, kde poslední obsahuje prompt od uživatele
        user_prompt = messages[-1]["content"]

        # inicializace vLLM klienta pro inference

        # generování textu

        if self.use_ollama:
            chat_res = self.client.chat.completions.create(
                model=self.ollama_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return chat_res.choices[0].message.content.strip()

        else:
            outputs = self.client.generate(
                [{"prompt": user_prompt}],
                sampling_params=SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            )
            return outputs[0].outputs[0].text