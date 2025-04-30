# paper -  https://arxiv.org/pdf/2410.02832 
# kod   -  https://github.com/yueliu1999/FlipAttack


# attacks/Flip.py
class Flip:
    name = "flip"

    def __init__(self, cfg: dict):
        # zavolá se při importu/přípravě
        print(f"[Flip] initialized with config: {cfg}")

    def attack(text: str, **cfg) -> dict:
        # cfg je váš params dict z YAMLu
        flip = Flip(cfg)
        return flip.run(text)
