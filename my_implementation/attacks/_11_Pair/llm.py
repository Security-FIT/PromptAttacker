from vllm import LLM as VLLMClient, SamplingParams
from typing import List, Dict
import sys 

class LLM:
    """
    Ultra-lehký wrapper nad vLLM pro lokální inference.
    Přidána podpora pro sekvenční načítání/uvolňování modelů.
    Metoda 'response' nyní vrací přímo vygenerovaný text (string).
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
        seed: int = 42,
        max_model_len: int = 32768,
    ) -> None:
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.client = None  # Model se nenačítá při inicializaci
        self.params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            top_k=-1,
            seed=seed,
            stop=[]
        )

    def load_model(self) -> None:
        """Načte model do paměti. Volá se pouze, pokud ještě není načten."""
        if self.client is None:
            print(f"[INFO] Načítám model: {self.model_path} s max_model_len={self.max_model_len}")
            # Důležité: Předáváme max_model_len do VLLMClient pro správnou konfiguraci
            self.client = VLLMClient(model=self.model_path)
        else:
            print(f"[INFO] Model {self.model_path} je již načten.")

    def unload_model(self) -> None:
        """Uvolní model z paměti."""
        if self.client is not None:
            print(f"[INFO] Uvolňuji model: {self.model_path}")
            # 'del self.client' by měl stačit k uvolnění referencí a umožnit Garbage Collectoru uvolnit paměť.
            # del self.client
            self.client = None
        else:
            print(f"[INFO] Model {self.model_path} není načten.")

    # ------------------------------------------------------------------ #

    def response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generuje odpověď pomocí načteného modelu.
        Vrací přímo vygenerovaný text (string).
        """
        if self.client is None:
            raise RuntimeError(f"Model {self.model_path} není načten. Zavolejte .load_model() před .response().")

        prompt = self._to_prompt(messages)
        
        generations = self.client.generate(prompt, self.params)
        
        if generations and generations[0] and generations[0].outputs:
            return generations
        else:
            # Pokud není nalezen očekávaný text, vrátí prázdný string a vypíše upozornění.
            print(f"[UPOZORNĚNÍ] Generování nevrátilo očekávaný text pro prompt: {prompt[:100]}...", file=sys.stderr)
            return ""

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Převádí seznam zpráv do formátu promptu, který vLLM očekává.
        Používá Llama-2 chat formát s <s>[ROLE] content</s>.
        """
        out = []
        for m in messages:
            role = m["role"].upper() # Převede roli na velká písmena (např. 'user' na 'USER')
            out.append(f"<s>[{role}] {m['content']}</s>")
        return "\n".join(out)