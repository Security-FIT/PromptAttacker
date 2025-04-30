

class Evaluator:
    name = "evaluator"

    def __init__(self, cfg: dict):
        print(f"[Evaluator] initialized with config: {cfg}")

    def evaluate(self, prompt: str, results: dict) -> dict:
        print(f"[Evaluator] evaluate() got prompt: {prompt}")
        print(f"[Evaluator] evaluate() got results: {results}")
        # vrátíme dummy metriku
        return {"dummy_score": 0.0}