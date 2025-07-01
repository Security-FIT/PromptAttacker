# attacks/_4_ReNeLLM/llm.py
from vllm import LLM as VLLMClient, SamplingParams

class LLM:
    def __init__(self, model_path: str, temperature=0.0, max_tokens=512):
        self.client = VLLMClient(model=model_path, trust_remote_code=True)
        self.params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens)

    def response(self, messages):
        prompt = messages[-1]["content"]
        out = self.client.generate([{"prompt": prompt}], self.params)
        return out[0].outputs[0].text.strip()
