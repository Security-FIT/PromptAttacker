#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_empty_files_hardcoded.py

Prohledá pevně zadaný adresář a vypíše soubory, které jsou prázdné.
Konfigurace je napevno v proměnných níže.
"""

from pathlib import Path
from typing import Iterator, List
import os

# ---------------------- NASTAVENÍ (upravit podle potřeby) ---------------------- #
ROOT_DIR = "/storage/brno2/home/xkaska01/master/my_implementation/results"   # <-- sem vlož absolutní (nebo relativní) cestu, kterou chceš prohledat
RECURSIVE = True                 # True = prohledat rekurzivně (podsložky), False = jen aktuální adresář
TREAT_WHITESPACE_AS_EMPTY = False  # True považuje soubory obsahující pouze whitespace za prázdné (pomalejší)
RELATIVE_OUTPUT = False          # True vypíše cesty relativně vůči ROOT_DIR, jinak plné cesty
# ------------------------------------------------------------------------------- #

def iter_files(root: Path, recursive: bool = True) -> Iterator[Path]:
    if recursive:
        for p in root.rglob('*'):
            if p.is_file():
                yield p
    else:
        for p in root.iterdir():
            if p.is_file():
                yield p

def is_zero_byte(p: Path) -> bool:
    try:
        return p.stat().st_size == 0
    except Exception:
        return False

def is_whitespace_only(p: Path) -> bool:
    # Pokusíme se otevřít jako text; pokud to nejde (binární), považujeme za non-whitespace
    try:
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            for chunk in iter(lambda: f.read(8192), ''):
                if chunk and chunk.strip():
                    return False
            return True
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
            if treat_whitespace_as_empty:
                # pouze testujeme soubory, které nejsou 0B
                if is_whitespace_only(p):
                    out.append(p)
        except Exception:
            # ignorujeme soubory, ke kterým nemáme přístup
            continue
    return out

def main():
    root = Path(ROOT_DIR).resolve()
    if not root.exists():
        print(f"Chyba: cesta neexistuje: {root}")
        return
    found = find_empty_files(root, recursive=RECURSIVE, treat_whitespace_as_empty=TREAT_WHITESPACE_AS_EMPTY)

    if not found:
        print("✅ Nenašel jsem žádné prázdné soubory.")
        return

    print(f"🔍 Našel jsem {len(found)} prázdných souborů:")
    for p in found:
        if RELATIVE_OUTPUT:
            try:
                print(str(p.relative_to(root)))
            except Exception:
                print(str(p))
        else:
            print(str(p))

if __name__ == "__main__":
    main()
