## @file base_attack.py
#  @brief Minimal base class for prefix/suffix-based prompt modification attacks
#
#  This file defines a generic BaseAttack and its corresponding configuration
#  dataclass. The attack implements the simplest form of prompt manipulation by
#  prepending and/or appending fixed strings to the last user message.
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BaseAttackConfig:
    attacker_cls: str = field(default="BaseAttack", init=False)
    attacker_name: str = field(default="Base", init=False)

    prefix: str = ""
    suffix: str = ""


class BaseAttack:

    def __init__(self, cfg: BaseAttackConfig) -> None:
        self.prefix = cfg.prefix
        self.suffix = cfg.suffix

    def attack(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> List[Dict[str, str]]:

        if not messages:
            raise ValueError("`messages` nesmí být prázdné.")

        user_content = messages[-1]["content"]
        modified = f"{self.prefix}{user_content}{self.suffix}"

        messages[-1]["content"] = modified
        return messages
