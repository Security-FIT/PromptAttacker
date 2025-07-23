# TENTO FILE JE MUUUUUUUUUUJ

from vllm import LLM as VLLMClient, SamplingParams

class LLM:
    """
    Jednoduchý vLLM wrapper s offload/swap/quantization.
    """

    def __init__(self,
                 model_path: str,
                 temperature: float,
                 max_tokens: int,
                 engine_url: str | None       = None,
                 tensor_parallel_size: int    = 1,
                 swap_space: int             = 16,   # 16 GB na CPU
                 gpu_memory_utilization: float = 0.8,       # nevyplnit kartu zcela
                 quantization: str | None     = None):      # "bitsandbytes" pro 8-bit
        self.model_path           = model_path
        self.temperature          = temperature
        self.max_tokens           = max_tokens
        self.engine_url           = engine_url
        self.tp_size              = tensor_parallel_size
        self.swap_space           = swap_space
        self.gpu_mem_util         = gpu_memory_utilization
        self.quantization         = quantization

        self.client = VLLMClient(model=self.model_path)


    def response(self, messages: list[dict]) -> str:
        prompt = messages[-1]["content"]


        outs = self.client.generate(
            [{"prompt": prompt}],
            sampling_params=SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        )
        return outs[0].outputs[0].text
