from vllm import LLM as VLLMClient, SamplingParams

class LLM:
    """
    Velmi jednoduchý wrapper kolem vLLM pro lokální inference.

    Příklad:
        victim = LLM(
            model_path="gpt-4o-mini",
            temperature=0.0,
            max_tokens=512
        )
        response = victim.response([{"role":"user","content":"nějaký prompt"}])
    """
    def __init__(self, model_path: str, temperature: float = 0.8, max_tokens: int = 512):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = VLLMClient(
            model=self.model_path,
            trust_remote_code=True,
        )

    def response(self, messages: list[dict]) -> str:
        """
        messages: seznam zpráv s poli 'role' a 'content'. Poslední zpráva
                  obsahuje user‐prompt (zakódovaný řetězec).
        Vrací odpověď modelu (text output).
        """
        user_prompt = messages[-1]["content"]



        outputs = self.client.generate(
            [{"prompt": user_prompt}],
            sampling_params=SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        )

        # vLLM vrací výsledky v .outputs
        return outputs[0].outputs[0].text
