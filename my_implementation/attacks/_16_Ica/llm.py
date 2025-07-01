# attacks/_3_ICA/llm.py
from vllm import LLM as VLLMClient, SamplingParams

class LLM:
    def __init__(self, model_path: str,
                 temperature: float = 0.8,
                 max_tokens: int = 512):
        self.model_path  = model_path
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def response(self, messages: list[dict]) -> str:
        prompt = messages[-1]["content"]  # ICA má všechno v user-promptu

        client = VLLMClient(
            model=self.model_path,
            trust_remote_code=True,
        )

        out = client.generate(
            [{"prompt": prompt}],
            sampling_params=SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
        )
        return out[0].outputs[0].text
