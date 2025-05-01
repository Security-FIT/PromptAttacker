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
        # cesta k lokálnímu HF modelu
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        # vytvoříme klienta až v response(), abychom mohli znovu použít wrapper

    def response(self, messages: list[dict]) -> str:
        # očekáváme seznam zpráv, kde poslední obsahuje prompt od uživatele
        user_prompt = messages[-1]["content"]

        # inicializace vLLM klienta pro inference
        client = VLLMClient(
            model=self.model_path
        )

        # generování textu
        outputs = client.generate(
            [{"prompt": user_prompt}],
            sampling_params=SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        )

        # v nové verzi vLLM jsou výstupy v outputs[0].outputs
        return outputs[0].outputs[0].text
