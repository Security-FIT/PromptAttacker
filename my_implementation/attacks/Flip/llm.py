# llm.py
import os
from vllm import LLM as VLLM, SamplingParams

class LLM:
    """
    Wrapper around vLLM to load a local model and generate responses.
    """
    def __init__(self, model_path, temperature=0, max_token=None, retry_time=None, failed_sleep_time=None, round_sleep_time=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        # Load local model via vLLM
        self.llm = VLLM(model=model_path)
        # Configure sampling parameters
        sampling_kwargs = {"temperature": temperature}
        if max_token and max_token > 0:
            sampling_kwargs["max_tokens"] = max_token
        self.sampling_params = SamplingParams(**sampling_kwargs)

    def response(self, messages):
        # `messages` is a list of dicts with 'content' keys; concatenate them into a prompt
        prompt = "\n".join([m['content'] for m in messages])
        # Generate locally
        outputs = list(self.llm.generate([prompt], sampling_params=self.sampling_params))
        return outputs[0].text
