import csv
import os
from typing import Dict, Tuple, Set

# --- Konfigurace ---
INPUT_CSV_FILE = '/storage/brno2/home/xkaska01/master/my_implementation/dataset/cysecbench_adv_small.csv' 

# Definice řádkových intervalů pro každý původní dataset
# Klíče jsou názvy tvých sub-datasetů, hodnoty jsou tuple (počáteční_řádek, koncový_řádek)
# POZOR: Tyto řádky se vztahují k "číslování řádků v CSV souboru",
#       přičemž se předpokládá HLAVIČKA na řádku 1.
#       Pokud nemáš hlavičku, začni intervaly od 1!
#       Např. (2, 501) znamená řádky 2, 3, ..., 501.
ORIGINAL_DATASET_ROW_RANGES = {
    'cysecbench': (2, 501),   # Řádky 2 až 501
    'cypher': (502, 600),     # Řádky 502 až 600
    'muj_dataset': (601, 1120), # Řádky 601 až 1120
}

# --- Funkce pro počítání unikátních targetů ---
def count_unique_targets(input_file: str, row_ranges: Dict[str, Tuple[int, int]]):
    """
    Spočítá a vypíše počet unikátních 'target' hodnot
    pro každý definovaný sub-dataset a celkově.
    """
    total_unique_targets_all_data = set()
    strata_unique_targets: Dict[str, Set[str]] = {name: set() for name in row_ranges.keys()}
    
    try:
        with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            
            original_header = next(reader) # Přeskoč hlavičku (řádek 1)
            print(f"Načtená původní hlavička: {original_header}\n")

            # Kontrola, zda sloupec 'target' existuje a je na očekávaném indexu (1)
            if len(original_header) < 2:
                print("Chyba: CSV soubor nemá dostatek sloupců. Očekává se 'target' ve druhém sloupci (index 1).")
                return

            for i, row in enumerate(reader):
                # i je 0-index pro datové řádky po hlavičce
                # actual_csv_row_num je skutečné číslo řádku v CSV souboru (počínaje 2 pro data)
                actual_csv_row_num = i + 2 
                
                if len(row) > 1: # Ujistíme se, že řádek má alespoň dva sloupce pro 'target'
                    target_value = row[1] # 'target' je na indexu 1 (druhý sloupec)
                    total_unique_targets_all_data.add(target_value) # Přidáme do celkového počítadla

                    # Přiřadíme target k příslušné stratě
                    found_stratum = False
                    for stratum_name, (start_row, end_row) in row_ranges.items():
                        if start_row <= actual_csv_row_num <= end_row:
                            strata_unique_targets[stratum_name].add(target_value)
                            found_stratum = True
                            print(strata_unique_targets)

                            break
                    
                    if not found_stratum:
                        # Může se stát, že řádek je mimo definované rozsahy, ale to je v pořádku.
                        pass 
                else:
                    print(f"Varování: Řádek {actual_csv_row_num} neobsahuje dostatek sloupců pro 'target'. Bude ignorován pro počítání targetů.")

        print("--- Výsledky počítání unikátních 'Target' hodnot ---")
        
        # Výpis pro každou stratu
        for stratum_name, unique_targets_set in strata_unique_targets.items():
            print(f"Dataset '{stratum_name}': {len(unique_targets_set)} unikátních 'target' hodnot.")
        
        # Celkový výpis
        print(f"\nCelkem všech unikátních 'target' hodnot v celém souboru: {len(total_unique_targets_all_data)}.")

    except FileNotFoundError:
        print(f"Chyba: Vstupní soubor '{input_file}' nebyl nalezen.")
    except Exception as e:
        print(f"Došlo k chybě při čtení nebo zpracování souboru: {e}")

# --- Spuštění skriptu ---
if __name__ == "__main__":
    # PŘED SPUŠTĚNÍM:
    # 1. Ujistěte se, že 'INPUT_CSV_FILE' ukazuje na váš velký CSV soubor.
    #    Předpokládá se, že 'goal' je v prvním sloupci (index 0)
    #    a 'target' ve druhém sloupci (index 1).
    # 2. Zkontrolujte 'ORIGINAL_DATASET_ROW_RANGES' a ujistěte se, že řádkové intervaly
    #    přesně odpovídají rozdělení ve vašem CSV a zda počítají s hlavičkou na řádku 1.
    
    count_unique_targets(INPUT_CSV_FILE, ORIGINAL_DATASET_ROW_RANGES)