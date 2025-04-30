# paper -  https://arxiv.org/pdf/2410.02832 
# kod   -  https://github.com/yueliu1999/FlipAttack


# attacks/Flip.py
class Flip:
    name = "flip"

    def __init__(self, cfg: dict):
        # zavolá se při importu/přípravě
        print(f"[Flip] initialized with config: {cfg}")

    def run(self, prompt: str) -> dict:
        # zavolá se pro každý prompt
        print(f"[Flip] run() got prompt: {prompt}")
        # jako výsledek vrátíme pro ukázku nějaký dict
        flipped = prompt[::-1]
        return {"flipped": flipped}
