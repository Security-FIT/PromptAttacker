# TENTO FILE JE MUUUUUUUUUUJ


from vllm import LLM as VLLMClient, SamplingParams

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
                 max_tokens: int = 512
                 ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens


    def response(self, messages: list[dict]) -> str:
        user_prompt = messages[-1]["content"]

        client = VLLMClient(
            model=self.model_path,
            trust_remote_code=True,  
        )

        outputs = client.generate(
            [{"prompt": user_prompt}],
            sampling_params=SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        )

        return outputs[0].outputs[0].text
