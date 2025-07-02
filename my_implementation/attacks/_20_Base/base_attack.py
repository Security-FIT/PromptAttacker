from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BaseAttackConfig:
    """
    Konfigurace pro BaseAttack.

    - prefix:  řetězec přidaný před původní user-prompt
    - suffix:  řetězec přidaný za původní user-prompt
    """
    attacker_cls: str = field(default="BaseAttack", init=False)
    attacker_name: str = field(default="Base", init=False)

    prefix: str = ""
    suffix: str = ""


class BaseAttack:
    """
    Jednoduchý útok, který vezme poslední uživatelský prompt (messages[-1])
    a obalí jej prefixem/suffixem z konfigurace.
    """

    def __init__(self, cfg: BaseAttackConfig) -> None:
        self.prefix = cfg.prefix
        self.suffix = cfg.suffix

    # hlavní API – stejné jméno i signatura, jakou používají ostatní útoky
    def attack(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> List[Dict[str, str]]:
        """
        Vrátí upravený seznam zpráv, kde poslední user-zpráva je obalená
        přednastaveným prefixem/suffixem.
        """
        if not messages:
            raise ValueError("`messages` nesmí být prázdné.")

        # předpokládáme, že poslední položka je uživatel
        user_content = messages[-1]["content"]
        modified = f"{self.prefix}{user_content}{self.suffix}"

        # zapíšeme zpět
        messages[-1]["content"] = modified
        return messages
