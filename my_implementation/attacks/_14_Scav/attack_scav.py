# encoding: utf-8
"""
SCAV Attacker – auto-switch podle „8b_or_70b“
============================================
Lookup-based útočník, který nahrazuje poslední user prompt staticky
optimalizovanou instrukcí z CSV:

    original_instruction,best_instruction,initial_prob,final_prob,fitness

CSV pro 8 B a 70 B variantu se vybírá klíčem **8b_or_70b** v configu.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from attacks.helpers import is_user_turn, BaseAttacker, BaseAttackerConfig


# --------------------------------------------------------------------------- #
#                               1. Config dataclass                           #
# --------------------------------------------------------------------------- #
@dataclass
class ScavAttackerConfig(BaseAttackerConfig):
    """Konfigurace SCAV útočníka; vše lze přepsat v YAML."""

    attacker_cls: str = field(default="ScavAttacker")
    attacker_name: str | None = field(default="Scav")

    # „8b“ nebo „70b“ (přebíjí automatický odhad z názvu modelu)
    model_size: str = field(default="8b",
                            metadata={"help": '"8b" nebo "70b"'})

    # Cesty k optimalizovaným CSV
    optimized_dataset_8b: str = field(
        default="data/scav/optimized_instructions_8b.csv")
    optimized_dataset_70b: str = field(
        default="data/scav/optimized_instructions_70b.csv")

    # starší názvy (pro kompatibilitu)
    optimzed_file_8b: str = field(default="", repr=False)
    optimzed_file_70b: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        # fallback na legacy klíče, pokud jsou vyplněné
        if self.optimzed_file_8b:
            self.optimized_dataset_8b = self.optimzed_file_8b
        if self.optimzed_file_70b:
            self.optimized_dataset_70b = self.optimzed_file_70b


# --------------------------------------------------------------------------- #
#                                 2. Attacker                                 #
# --------------------------------------------------------------------------- #
class ScavAttacker(BaseAttacker):
    """Lookup útočník: nahradí prompt, pokud existuje optimalizovaná varianta."""

    def __init__(self, config: ScavAttackerConfig):
        super().__init__(config)
        self.cfg = config
        self._df: pd.DataFrame | None = None  # lazy-loaded cache

    # ----------------------------- interní helpery ------------------------ #
    @staticmethod
    def _load_csv(path: str) -> pd.DataFrame:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[Scav] CSV nenalezeno: {path}")
        # vezmeme jen první dvě kolony (ostatní hodnoty ignorujeme)
        return pd.read_csv(
            path, header=None, usecols=[0, 1],
            names=["original", "best"], engine="python"
        )

    def _get_df(self) -> pd.DataFrame:
        """Vrátí správný DataFrame podle cfg.model_size."""
        if self._df is not None:
            return self._df

        path = (self.cfg.optimized_dataset_8b
                if self.cfg.model_size.lower().startswith("8")
                else self.cfg.optimized_dataset_70b)
        self._df = self._load_csv(path)
        return self._df

    def _lookup(self, original: str) -> str | None:
        df = self._get_df()
        row = df[df["original"] == original]
        return None if row.empty else str(row.iloc[0]["best"])

    # ------------------------------- public API --------------------------- #
    def attack(self, messages: List[Dict[str, str]], **__) -> List[Dict[str, str]]:
        """Pokud najdeme optimalizaci, nahradíme obsah posledního user turnu."""
        assert is_user_turn(messages)

        original_prompt = messages[-1]["content"]
        best_prompt = self._lookup(original_prompt)

        if best_prompt and best_prompt.strip():
            messages[-1]["content"] = best_prompt

        return messages
