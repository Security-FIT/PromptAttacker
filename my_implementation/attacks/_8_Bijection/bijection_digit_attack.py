# attacks/Digit/digit_attack.py

import random

class DigitAttack:
    """
    Třída pro digitální bijekci (0-9). Konstruktor vygeneruje náhodné mapování
    mezi ciframi 0-9. Funkce encode/decode pak umí kódovat a dekódovat libovolný text.
    """

    def __init__(self):
        # 1) Generujeme náhodnou bijekci pro cifry '0'-'9'
        digits = list("0123456789")
        shuffled = digits.copy()
        random.shuffle(shuffled)
        self.bijection = {d: s for d, s in zip(digits, shuffled)}
        # Inverzní slovník pro dekódování
        self.de_bijection = {s: d for d, s in self.bijection.items()}

    def encode(self, text: str) -> str:
        """
        Projde každý znak v textu a pokud je to číslice 0-9, nahradí ji podle bijekce.
        Jinak ponechá znak beze změny.
        """
        return "".join(self.bijection[ch] if ch in self.bijection else ch for ch in text)

    def decode(self, text: str) -> str:
        """
        Zpětná operace: každou cifru v textu vrátí do původní podoby podle inverzní bijekce.
        Ostatní znaky ponechá beze změny.
        """
        return "".join(self.de_bijection[ch] if ch in self.de_bijection else ch for ch in text)

    def generate(self, original_prompt: str) -> dict:
        """
        Vrátí informace nutné pro útok:
          - 'original_prompt': původní cíl (řetězec z CSV sloupce 'goal').
          - 'encoded_prompt': zakódovaný prompt, kam byly všechny cifry 0-9
                                nahrazeny jinými ciframi podle náhodné bijekce.
          - 'bijection': kopie použité bijekce (pro debug/rozbor, není nutné zapisovat do výsledku).
        """
        encoded = self.encode(original_prompt)
        return {
            "original_prompt": original_prompt,
            "encoded_prompt": encoded,
            "bijection": self.bijection.copy()
        }
