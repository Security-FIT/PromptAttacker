from vllm import LLM as VLLMClient, SamplingParams

class LLM:
    """
    Very simple wrapper around vLLM for local inference.
    """
    def __init__(self, model_path: str, temperature: float = 0.8, max_tokens: int = 512):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = VLLMClient(model=self.model_path, trust_remote_code=True)

    def response(self, messages: list[dict]) -> str:
        user_prompt = messages[-1]['content']
        outputs = self.client.generate(
            [{"prompt": user_prompt}],
            sampling_params=SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        )
        return outputs[0].outputs[0].text