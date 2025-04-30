
class DefenseEA:
    name = "defense"

    def __init__(self, cfg: dict):
        print(f"[DefenseEA] initialized with config: {cfg}")

    def run(self, prompt: str) -> dict:
        print(f"[DefenseEA] run() got prompt: {prompt}")
        # zatím jen pass-through
        return {"original": prompt, "defended": prompt}