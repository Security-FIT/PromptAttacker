import csv
import os
import random
from typing import List, Dict, Any, Tuple

# --- Konfigurace ---
INPUT_CSV_FILE = '/storage/brno2/home/xkaska01/master/my_implementation/dataset/cysecbench_adv_small.csv' 
OUTPUT_CSV_FILE = 'dataset_40_prompts_proportional_unique_targets.csv' 

TARGET_PROMPTS_COUNT = 40 

# Definice řádkových intervalů pro každý původní dataset
# Klíče jsou názvy tvých sub-datasetů, hodnoty jsou tuple (počáteční_řádek, koncový_řádek)
# POZOR: Tyto řádky se vztahují k "číslování řádků v CSV souboru",
#       přičemž se předpokládá HLAVIČKA na řádku 1.
#       Pokud nemáš hlavičku, začni intervaly od 1!
#       Např. (2, 501) znamená řádky 2, 3, ..., 501.
ORIGINAL_DATASET_ROW_RANGES = {
    'cysecbench': (2, 501),   # Řádky 2 až 501 (500 promptů)
    'cypher': (502, 600),     # Řádky 502 až 600 (99 promptů)
    'muj_dataset': (601, 1120), # Řádky 601 až 1120 (560 promptů)
}

# --- PROPORČNÍ KVÓTY PRO ROVNOMĚRNÉ ZASTOUPENÍ ---
# Celkem promptů v původním fondu: 500 + 99 + 560 = 1159
# Proporce pro cílových 40 promptů:
# cysecbench: (500 / 1159) * 40 = 17.256 -> zaokrouhleno na 17
# cypher:     (99 / 1159) * 40 = 3.416  -> zaokrouhleno na 3
# muj_dataset: (560 / 1159) * 40 = 19.328 -> zaokrouhleno na 20
TARGET_QUOTAS_PER_SUB_DATASET = {
    'cysecbench': 17,
    'cypher': 3,
    'muj_dataset': 20,
}

# Kontrola součtu kvót
if sum(TARGET_QUOTAS_PER_SUB_DATASET.values()) != TARGET_PROMPTS_COUNT:
    raise ValueError(f"Součet pevných kvót ({sum(TARGET_QUOTAS_PER_SUB_DATASET.values())}) musí být roven TARGET_PROMPTS_COUNT ({TARGET_PROMPTS_COUNT})!")

print(f"Cílové PROPORČNÍ kvóty pro výběr {TARGET_PROMPTS_COUNT} promptů z každého sub-datasetu: {TARGET_QUOTAS_PER_SUB_DATASET}")
print(f"Intervaly pro rozdělení datasetů: {ORIGINAL_DATASET_ROW_RANGES}")

def create_dataset_B_from_csv(input_file: str, output_file: str, row_ranges: Dict[str, Tuple[int, int]], quotas: Dict[str, int]):
    """
    Vytvoří Dataset B o 40 promptech stratifikovaným náhodným výběrem
    na základě řádkových intervalů a pevných (proporčních) kvót.
    Prioritizuje výběr unikátních "target" hodnot z každého sub-datasetu.
    Vypíše původní řádky a jejich zdroj pro každý vybraný prompt.
    """
    all_rows_with_original_info = [] 
    original_header = []

    try:
        with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            
            original_header = next(reader) # Načti hlavičku (řádek 1)
            print(f"Načtená původní hlavička: {original_header}")

            # Načti všechny datové řádky spolu s jejich původním číslem řádku v CSV
            for i, row in enumerate(reader):
                # i je 0-index pro datové řádky po hlavičce
                # i + 2 je skutečné číslo řádku v CSV souboru (pokud je hlavička na řádku 1)
                all_rows_with_original_info.append((row, i + 2)) 
        
        print(f"Načteno {len(all_rows_with_original_info)} datových řádků z '{input_file}'.")

    except FileNotFoundError:
        print(f"Chyba: Vstupní soubor '{input_file}' nebyl nalezen.")
        return
    except Exception as e:
        print(f"Došlo k chybě při čtení souboru: {e}")
        return

    # Rozděl řádky do strat podle zadaných intervalů
    strata_data: Dict[str, List[Tuple[List[str], int]]] = {name: [] for name in row_ranges.keys()}
    
    for row_data, original_row_num in all_rows_with_original_info:
        found_stratum = False
        for stratum_name, (start_row, end_row) in row_ranges.items():
            if start_row <= original_row_num <= end_row:
                strata_data[stratum_name].append((row_data, original_row_num))
                found_stratum = True
                break
        
        if not found_stratum:
            # print(f"Varování: Řádek {original_row_num} se neshoduje s žádným definovaným intervalem. Bude ignorován.")
            pass # Může být mnoho řádků mimo definované rozsahy, takže nebudeme spamovat konzoli

    # Ověř, zda máme data v každé stratě
    print("\n--- Rozložení promptů ve stratách před výběrem ---")
    for name, data_list in strata_data.items():
        # Zjištění počtu unikátních targetů v této stratě
        unique_targets_in_stratum = set()
        for row_data, _ in data_list:
            if len(row_data) > 1: # Předpokládáme, že target je ve druhém sloupci (index 1)
                unique_targets_in_stratum.add(row_data[1])
        print(f"Strata '{name}' obsahuje {len(data_list)} promptů a {len(unique_targets_in_stratum)} unikátních targetů.")
        if not data_list and quotas[name] > 0:
            print(f"  Upozornění: Strata '{name}' je prázdná, ale vyžaduje {quotas[name]} promptů.")

    selected_dataset_B_rows_with_info = [] # Ukládáme (řádek_dat, původní_číslo_řádku_v_CSV, název_straty)

    # Provedeme výběr z každé straty s preferencí unikátních targetů
    print("\n--- Probíhá výběr promptů pro Dataset B ---")
    for stratum_name, quota in quotas.items():
        available_prompts_in_stratum = strata_data.get(stratum_name, [])
        
        if not available_prompts_in_stratum:
            print(f"  Varování: Není dostatek promptů ve stratě '{stratum_name}' pro naplnění kvóty {quota}. Vybráno 0.")
            continue
        
        # 1. Krok: Vybereme co nejvíce unikátních targetů
        selected_targets_for_stratum = set()
        prompts_with_unique_targets = []
        remaining_prompts_for_stratum = []
        
        # Aby byl výběr unikátních targetů náhodný, promícháme dostupné prompte
        random.shuffle(available_prompts_in_stratum)

        for row_data, original_row_num in available_prompts_in_stratum:
            if len(row_data) > 1:
                target_value = row_data[1]
                if target_value not in selected_targets_for_stratum:
                    prompts_with_unique_targets.append((row_data, original_row_num))
                    selected_targets_for_stratum.add(target_value)
                else:
                    remaining_prompts_for_stratum.append((row_data, original_row_num))
            else:
                remaining_prompts_for_stratum.append((row_data, original_row_num)) # Prompty bez targetu jdou do zbytku

        # Vezmeme unikátní prompte, kolik jen můžeme, až do kvóty
        num_unique_selected = min(quota, len(prompts_with_unique_targets))
        selected_for_current_stratum = random.sample(prompts_with_unique_targets, num_unique_selected)
        
        # 2. Krok: Doplň zbývající místa z "remaining_prompts_for_stratum"
        num_remaining_to_select = quota - len(selected_for_current_stratum)
        
        if num_remaining_to_select > 0:
            if len(remaining_prompts_for_stratum) < num_remaining_to_select:
                print(f"  Upozornění: Nedostatek zbývajících promptů ve stratě '{stratum_name}' pro doplnění kvóty {quota}. Původní: {len(available_prompts_in_stratum)}, unikátní targety: {len(prompts_with_unique_targets)}, zbývající: {len(remaining_prompts_for_stratum)}. Bude vybráno méně než kvóta.")
                
            selected_for_current_stratum.extend(
                random.sample(remaining_prompts_for_stratum, min(num_remaining_to_select, len(remaining_prompts_for_stratum)))
            )
        
        # Přidej vybrané prompte (s jejich informacemi o původu) do celkového seznamu
        for row_data, original_row_num in selected_for_current_stratum:
            selected_dataset_B_rows_with_info.append((row_data, original_row_num, stratum_name))
        
        print(f"  Ze straty '{stratum_name}' vybráno {len(selected_for_current_stratum)} promptů (kvóta {quota}). Unikátních targetů: {num_unique_selected}.")


    random.shuffle(selected_dataset_B_rows_with_info)

    final_selected_rows_for_output = selected_dataset_B_rows_with_info[:TARGET_PROMPTS_COUNT]

    if len(final_selected_rows_for_output) != TARGET_PROMPTS_COUNT:
        print(f"Upozornění: Celkový počet vybraných promptů ({len(final_selected_rows_for_output)}) se liší od cílového ({TARGET_PROMPTS_COUNT}). To může nastat, pokud je nedostatek dat pro naplnění kvót.")

    # --- Výpis původu promptů a ověření unikátnosti targetů ve výsledku ---
    print("\n--- Původ promptů ve výsledném Datasetu B ---")
    print(f"Dataset B obsahuje {len(final_selected_rows_for_output)} promptů:")
    
    final_targets_in_dataset_B = {} # Pro kontrolu unikátních targetů ve finálním datasetu
    
    for i, (row_data, original_row_num, stratum_name) in enumerate(final_selected_rows_for_output):
        goal = row_data[0] if len(row_data) > 0 else "N/A"
        target = row_data[1] if len(row_data) > 1 else "N/A"
        
        if stratum_name not in final_targets_in_dataset_B:
            final_targets_in_dataset_B[stratum_name] = set()
        final_targets_in_dataset_B[stratum_name].add(target)

        print(f"  [{i+1}/{TARGET_PROMPTS_COUNT}] Z původního řádku CSV: {original_row_num} | Původní dataset: '{stratum_name}' | Goal: '{goal}' | Target: '{target}'")

    print("\n--- Počet unikátních 'Target' hodnot ve výsledném Datasetu B dle straty ---")
    for stratum, targets_set in final_targets_in_dataset_B.items():
        print(f"  Strata '{stratum}': {len(targets_set)} unikátních targetů.")


    # Zápis do nového CSV souboru se stejnou hlavičkou
    try:
        # Extrahujeme jen datové řádky pro zápis
        rows_to_write = [item[0] for item in final_selected_rows_for_output]
        
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(original_header) # Zápis původní hlavičky
            writer.writerows(rows_to_write) # Zápis vybraných datových řádků

        print(f"\nÚspěšně vytvořen Dataset B s {len(rows_to_write)} prompte.")
        print(f"Výsledný soubor uložen do: {os.path.abspath(output_file)}")

    except Exception as e:
        print(f"Došlo k chybě při zápisu souboru: {e}")


# --- Spuštění skriptu ---
if __name__ == "__main__": 
    # PŘED SPUŠTĚNÍM:
    # 1. Ujistěte se, že 'INPUT_CSV_FILE' ukazuje na váš velký CSV soubor.
    #    Tento soubor by měl mít 'goal' v prvním sloupci (index 0)
    #    a 'target' ve druhém sloupci (index 1).
    # 2. Zkontrolujte 'ORIGINAL_DATASET_ROW_RANGES' a ujistěte se, že řádkové intervaly
    #    přesně odpovídají rozdělení ve vašem CSV a zda počítají s hlavičkou na řádku 1.
    #    (Řádek 1 je hlavička, datové řádky začínají od 2.)
    # 3. Zkontrolujte 'TARGET_QUOTAS_PER_SUB_DATASET' pro požadované proporcionální rozložení 40 promptů.
    
    create_dataset_B_from_csv(
        INPUT_CSV_FILE,
        OUTPUT_CSV_FILE,
        ORIGINAL_DATASET_ROW_RANGES,
        TARGET_QUOTAS_PER_SUB_DATASET
    )