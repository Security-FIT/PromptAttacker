import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 👉 TODO: sem dej cestu k tvému folderu s CSV soubory
FOLDER_PATH = "/storage/brno2/home/xkaska01/master/my_implementation/results/stats_without_defense"

# načteme všechny CSV soubory ve složce
csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))

if not csv_files:
    raise FileNotFoundError(f"Ve složce {FOLDER_PATH} jsem nenašel žádné .csv soubory.")

all_rows = []

for file_path in csv_files:
    # jméno modelu vezmeme z názvu souboru {model}.csv
    model_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Načítám {file_path} jako model '{model_name}'")

    df = pd.read_csv(file_path)

    # najdeme všechny score_* sloupce
    score_cols = [c for c in df.columns if c.startswith("score_")]
    if not score_cols:
        raise ValueError(f"V souboru {file_path} jsem nenašel žádné sloupce score_*")

    if "attack_type" not in df.columns:
        raise ValueError(f"V souboru {file_path} chybí sloupec 'attack_type'")

    # spočítáme průměr přes všechny score_* sloupce pro každý řádek
    df["mean_score"] = df[score_cols].mean(axis=1)

    # pro případ, že je víc řádků se stejným attack_type -> zprůměrujeme
    df_attack = (
        df.groupby("attack_type")["mean_score"]
        .mean()
        .reset_index()
    )
    df_attack["model"] = model_name

    all_rows.append(df_attack)

# spojíme všechny modely dohromady
combined = pd.concat(all_rows, ignore_index=True)

# vytvoříme matici model × attack_type
heatmap_data = combined.pivot(index="model", columns="attack_type", values="mean_score")

# pro jistotu seřadíme modely i útoky (nepovinné)
#  Seřazení heatmapy podle hodnot (nejvyšší vlevo nahoře, nejnižší vpravo dole)

# seřadíme modely podle průměrné hodnoty (Y osa – od nejvyšší po nejnižší)
model_order = heatmap_data.mean(axis=1).sort_values(ascending=False).index

# seřadíme útoky podle průměrné hodnoty (X osa – od nejvyšší po nejnižší)
attack_order = heatmap_data.mean(axis=0).sort_values(ascending=False).index

# aplikujeme řazení
heatmap_data = heatmap_data.loc[model_order, attack_order]


print("Tvar výsledné matice:", heatmap_data.shape)
# heatmap_data = heatmap_data.copy()
# heatmap_data[heatmap_data > 5] = heatmap_data[heatmap_data > 5] - 1
# heatmap_data[heatmap_data < 5] = heatmap_data[heatmap_data < 5] - 1

# vykreslení heatmapy pomocí matplotlib
fig, ax = plt.subplots(figsize=(max(8, 0.5 * heatmap_data.shape[1]),
                                max(6, 0.5 * heatmap_data.shape[0])))

im = ax.imshow(
    heatmap_data.values,
    aspect="auto",
    vmin=0,
    vmax=10,
    cmap="RdYlGn_r"
)

# osy
ax.set_xticks(np.arange(heatmap_data.shape[1]))
ax.set_xticklabels(heatmap_data.columns, rotation=90)

ax.set_yticks(np.arange(heatmap_data.shape[0]))
ax.set_yticklabels(heatmap_data.index)

ax.set_xlabel("Attack type")
ax.set_ylabel("Model")
ax.set_title("Heatmap průměrných score (model × attack_type)")

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data.values[i, j]
        if not np.isnan(val):
            ax.text(
                j, i,
                f"{val:.1f}",          
                ha="center",
                va="center",
                color="black",
                fontsize=9
            )


# barevná škála
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Mean score (0–10)")
cbar.ax.invert_yaxis()   # ← 10 nahoře, 0 dole
plt.tight_layout()

# uložení do souboru
plt.savefig("stats_experimental_without_defense11111.png", dpi=300, bbox_inches="tight")

# případně zobrazit:
# plt.show()

TXT_OUT = "heatmap_models_vs_attacks_stats_experimental_without_defe.txt"

with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("=== NUMERICKÁ HEATMAPA (Mean score, 0–10) ===\n\n")
    f.write(heatmap_data.round(3).to_string())
    f.write("\n\n")

    f.write("=== PRŮMĚR PŘES MODELY (attack_type) ===\n\n")
    f.write(heatmap_data.mean(axis=0).sort_values(ascending=False).round(3).to_string())
    f.write("\n\n")

    f.write("=== PRŮMĚR PŘES ÚTOKY (model) ===\n\n")
    f.write(heatmap_data.mean(axis=1).sort_values(ascending=False).round(3).to_string())
    f.write("\n")

print(f"[INFO] Numerická heatmapa uložena do: {TXT_OUT}")