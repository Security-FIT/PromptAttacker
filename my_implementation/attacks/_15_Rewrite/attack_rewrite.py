# attacks/_4_Rewrite/rewrite_attack.py
from typing import List, Dict

class RewriteAttack:
    """
    Připraví prompt, který donutil language-model prompt přepsat
    (technika „Rewrite“ – viz původní kód).

    Použití:
        attack = RewriteAttack(rewrite_template=MY_TEMPLATE)
        log, prompts = attack.generate("HOW TO MAKE A BOMB?")
    """

    def __init__(self, rewrite_template: str) -> None:
        self.rewrite_template = rewrite_template

    # ——— hlavní vstup ————————————————————————————————
    def generate(self, goal: str) -> tuple[str, List[Dict[str, str]]]:
        """
        Vrací (log, prompts) stejně jako FlipAttack i DeepInception,
        aby šel snadno zaměnit ve zbytku pipeline.

        Parameters
        ----------
        goal : str
            Řádek `goal` z datasetu.

        Returns
        -------
        log : str
            Protokol „TASK is '...'“ (stejně jako jinde).
        prompts : list[dict]
            Jediný user-prompt, kde je `goal` vložen do šablony.
        """
        attack_prompt = self.rewrite_template.format(content=goal)

        log = f"TASK is '{goal}'"
        prompts = [{"role": "user", "content": attack_prompt}]
        return log, prompts
