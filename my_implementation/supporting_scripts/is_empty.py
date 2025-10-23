#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_empty_files_hardcoded.py

Prohledá pevně zadaný adresář a:
 - vypíše soubory, které jsou prázdné,
 - vypíše soubory, které obsahují řetězec "[REQUEST ERROR]".

Konfigurace je napevno v proměnných níže.
"""

from pathlib import Path
from typing import Iterator, List
import os

# ---------------------- NASTAVENÍ (upravit podle potřeby) ---------------------- #
ROOT_DIR = "/storage/brno2/home/xkaska01/master/my_implementation/results"  # <-- sem vlož cestu
RECURSIVE = True                 # True = prohledat i podsložky
TREAT_WHITESPACE_AS_EMPTY = False
RELATIVE_OUTPUT = False
SEARCH_ERROR_STRING = "[REQUEST ERROR]"   # hledaný text v souborech
# ------------------------------------------------------------------------------- #

def iter_files(root: Path, recursive: bool = True) -> Iterator[Path]:
    if recursive:
        yield from (p for p in root.rglob('*') if p.is_file())
    else:
        yield from (p for p in root.iterdir() if p.is_file())

def is_zero_byte(p: Path) -> bool:
    try:
        return p.stat().st_size == 0
    except Exception:
        return False

def is_whitespace_only(p: Path) -> bool:
    try:
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            for chunk in iter(lambda: f.read(8192), ''):
                if chunk and chunk.strip():
                    return False
            return True
    except Exception:
        return False

def contains_error_string(p: Path, search_str: str) -> bool:
    """Vrátí True, pokud soubor obsahuje daný text (např. '[REQUEST ERROR]')."""
    try:
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if search_str in line:
                    return True
        return False
    except Exception:
        return False

def find_empty_files(root: Path,
                     recursive: bool = True,
                     treat_whitespace_as_empty: bool = False) -> List[Path]:
    out: List[Path] = []
    for p in iter_files(root, recursive=recursive):
        try:
            if is_zero_byte(p):
                out.append(p)
                continue
            if treat_whitespace_as_empty and is_whitespace_only(p):
                out.append(p)
        except Exception:
            continue
    return out

def find_error_files(root: Path,
                     recursive: bool = True,
                     search_str: str = SEARCH_ERROR_STRING) -> List[Path]:
    """Najde všechny soubory obsahující daný text."""
    out: List[Path] = []
    for p in iter_files(root, recursive=recursive):
        try:
            if contains_error_string(p, search_str):
                out.append(p)
        except Exception:
            continue
    return out

def main():
    root = Path(ROOT_DIR).resolve()
    if not root.exists():
        print(f"❌ Chyba: cesta neexistuje: {root}")
        return

    # --- Najdi prázdné soubory ---
    empty_files = find_empty_files(root, recursive=RECURSIVE, treat_whitespace_as_empty=TREAT_WHITESPACE_AS_EMPTY)

    # --- Najdi soubory s [REQUEST ERROR] ---
    error_files = find_error_files(root, recursive=RECURSIVE, search_str=SEARCH_ERROR_STRING)

    # --- Výpis ---
    if not empty_files:
        print("✅ Nenašel jsem žádné prázdné soubory.")
    else:
        print(f"🔍 Našel jsem {len(empty_files)} prázdných souborů:")
        for p in empty_files:
            print("  -", str(p.relative_to(root) if RELATIVE_OUTPUT else p))

    print()  # oddělení

    if not error_files:
        print("✅ Nenašel jsem žádné soubory obsahující '[REQUEST ERROR]'.")
    else:
        print(f"⚠️  Našel jsem {len(error_files)} souborů obsahujících '[REQUEST ERROR]':")
        for p in error_files:
            print("  -", str(p.relative_to(root) if RELATIVE_OUTPUT else p))

if __name__ == "__main__":
    main()
