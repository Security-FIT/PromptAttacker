import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 👉 TODO: sem dej cestu k tvému folderu s CSV soubory
FOLDER_PATH = "/storage/brno2/home/xkaska01/master/my_implementation/results/stats_without_defense"

# --- 1) Normalizace názvů útoků (kvůli "DIALOG COMPLETION" vs "DIALOG_COMPLETION", atd.) ---
def norm_attack_name(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s).strip().upper()
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

# --- 2) Mapování attack -> family podle tvého obrázku ---
attack_to_family = {
    "BASE": "Benchmark",

    "PAIR": "Black-box — LLM-generated",
    "TAP": "Black-box — LLM-generated",
    "PAST": "Black-box — LLM-generated",

    "SEQUENTIAL_ATTACK": "Black-box — Template-Completion — Scenario Nesting",
    "DIALOG_ATTACK": "Black-box — Template-Completion — Scenario Nesting",
    "DEEP_INCEPTION": "Black-box — Template-Completion — Scenario Nesting",
    "RENELLM": "Black-box — Template-Completion — Scenario Nesting",

    "ICA": "Black-box — Template-Completion — Context-Based",
    "CITATION": "Black-box — Template-Completion — Context-Based",
    "REWRITE": "Black-box — Template-Completion — Context-Based",
    "RANDOMSEARCH": "Black-box — Template-Completion — Context-Based",

    "CHAMELEON": "Black-box — Template-Completion — Code injection",
    "SQL": "Black-box — Template-Completion — Code injection",

    "GPT4CIPHER": "Black-box — Prompt Rewriting — Cipher",
    "CYPHER": "Black-box — Prompt Rewriting — Cipher",
    "FLIP": "Black-box — Prompt Rewriting — Cipher",
    "BIJECTION_RESULTS": "Black-box — Prompt Rewriting — Cipher",
    "ARTPROMPT": "Black-box — Prompt Rewriting — Cipher",

    "MULTILANG": "Black-box — Prompt Rewriting — Low-resource Language",
    "OVERLOAD": "Black-box — Prompt Rewriting — Low-resource Language",

    "SUFFIX": "White-box — Gradient-Based",
    "AUTODAN": "White-box — Gradient-Based",
}

# načteme všechny CSV soubory ve složce
csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"Ve složce {FOLDER_PATH} jsem nenašel žádné .csv soubory.")

all_rows_attack = []   # pro ranking konkrétních attacků
all_rows_family = []   # pro heatmapu po family

for file_path in csv_files:
    model_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Načítám {file_path} jako model '{model_name}'")

    df = pd.read_csv(file_path)

    score_cols = [c for c in df.columns if c.startswith("score_")]
    if not score_cols:
        raise ValueError(f"V souboru {file_path} jsem nenašel žádné sloupce score_*")
    if "attack_type" not in df.columns:
        raise ValueError(f"V souboru {file_path} chybí sloupec 'attack_type'")

    df["mean_score"] = df[score_cols].mean(axis=1)

    # normalizace + mapování na family
    df["attack_norm"] = df["attack_type"].apply(norm_attack_name)

    print("\n=== CO MAPUJI NA CO (attack_norm → family) ===")

    all_attacks_in_data = sorted(df["attack_norm"].unique())
    all_mapping_keys = sorted(attack_to_family.keys())

    print("\n--- ÚTOKY V DATECH (attack_norm) ---")
    for a in all_attacks_in_data:
        print(" ", a)

    print("\n--- KLÍČE V attack_to_family ---")
    for k in all_mapping_keys:
        print(" ", k)

    print("\n--- VÝSLEDEK MAPOVÁNÍ ---")
    for a in all_attacks_in_data:
        family = attack_to_family.get(a, "❌ UNKNOWN")
        print(f"{a:<25} → {family}")
    df["attack_family"] = df["attack_norm"].map(attack_to_family).fillna("UNKNOWN")

    # print("\n=== DEBUG: attack_type → attack_norm → attack_family ===")

    # mapping_debug = (
    #     df[["attack_type", "attack_norm", "attack_family"]]
    #     .drop_duplicates()
    #     .sort_values("attack_norm")
    # )

    # for _, row in mapping_debug.iterrows():
    #     print(
    #         f"RAW: {row['attack_type']:<25} "
    #         f"→ NORM: {row['attack_norm']:<20} "
    #         f"→ FAMILY: {row['attack_family']}"
    #     )
    # --- (A) průměr po konkrétním attack_type (pro ranking útoků) ---
    df_attack = (
        df.groupby("attack_norm")["mean_score"]
        .mean()
        .reset_index()
        .rename(columns={"attack_norm": "attack"})
    )
    df_attack["model"] = model_name
    all_rows_attack.append(df_attack)

    # --- (B) průměr po family (pro heatmapu) ---
    df_family = (
        df.groupby("attack_family")["mean_score"]
        .mean()
        .reset_index()
        .rename(columns={"attack_family": "family"})
    )
    df_family["model"] = model_name
    all_rows_family.append(df_family)

# spojení
combined_attack = pd.concat(all_rows_attack, ignore_index=True)
combined_family = pd.concat(all_rows_family, ignore_index=True)

# --- 3) Heatmapa model × family ---
heatmap_family = combined_family.pivot(index="model", columns="family", values="mean_score")

# řazení (stejně jako předtím)
model_order = heatmap_family.mean(axis=1).sort_values(ascending=False).index
family_order = heatmap_family.mean(axis=0).sort_values(ascending=False).index
heatmap_family = heatmap_family.loc[model_order, family_order]

print("Tvar výsledné matice (family):", heatmap_family.shape)

fig, ax = plt.subplots(figsize=(max(8, 0.7 * heatmap_family.shape[1]),
                                max(6, 0.5 * heatmap_family.shape[0])))

im = ax.imshow(heatmap_family.values, aspect="auto")

ax.set_xticks(np.arange(heatmap_family.shape[1]))
ax.set_xticklabels(heatmap_family.columns, rotation=90)
ax.set_yticks(np.arange(heatmap_family.shape[0]))
ax.set_yticklabels(heatmap_family.index)

ax.set_xlabel("Attack family")
ax.set_ylabel("Model")
ax.set_title("Heatmap průměrných score (model × attack_family)")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Mean score")

plt.tight_layout()
plt.savefig("heatmap_models_vs_attack_families.png", dpi=300, bbox_inches="tight")

# --- 4) Nejúspěšnější family + nejúspěšnější konkrétní attack ---
family_ranking = heatmap_family.mean(axis=0).sort_values(ascending=False)
print("\n=== Nejúspěšnější ATTACK FAMILY (průměr přes modely) ===")
print(family_ranking)

heatmap_attack = combined_attack.pivot(index="model", columns="attack", values="mean_score")
attack_ranking = heatmap_attack.mean(axis=0).sort_values(ascending=False)

print("\n=== Nejúspěšnější KONKRÉTNÍ ATTACK (průměr přes modely) ===")
print(attack_ranking)

# Volitelně: ulož rankingy do CSV
family_ranking.to_csv("ranking_attack_families.csv", header=["mean_score"])
attack_ranking.to_csv("ranking_attacks.csv", header=["mean_score"])
