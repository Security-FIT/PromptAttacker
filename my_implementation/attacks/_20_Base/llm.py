# attacks/_00_BaseAttack/llm.py
from vllm import LLM as VLLMClient, SamplingParams

class LLM:
    """
    Jednoduchý wrapper kolem vLLM (stejně jako u ICA/Flip).
    """
    def __init__(self, model_path: str, temperature: float = 0.0, max_tokens: int = 512):
        self.client = VLLMClient(model=model_path, trust_remote_code=True)
        self.params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def response(self, messages: list[dict]) -> str:
        """
        Volá poslední user‐prompt (messages[-1]) na model.
        """
        prompt = messages[-1]["content"]
        outputs = self.client.generate([{"prompt": prompt}], sampling_params=self.params)
        return outputs[0].outputs[0].text
