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

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
        seed: int = 42,
    ) -> None:
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.client = VLLMClient(model=model_path)
        self.params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            top_k=-1,
            logprobs=None,
            seed=seed,
            stop=[]
        )


    def response(self, messages: list[dict[str, str]]) -> str:
        prompt = self._messages_to_prompt(messages)
        generations = self.client.generate(prompt, self.params)
        return generations[0]


    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        """
        Triviální převod `[{"role":"system","content":...}, {"role":"user",...}]`
        na jediný textový prompt.
        """
        parts = []
        for m in messages:
            if m["role"] == "system":
                parts.append(f"<s>[SYSTEM] {m['content']}</s>")
            elif m["role"] == "user":
                parts.append(f"<s>[USER] {m['content']}</s>")
            else:
                raise ValueError(f"Unknown role {m['role']}")
        return "\n".join(parts)
