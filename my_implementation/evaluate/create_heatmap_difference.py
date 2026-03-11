import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SECOND_FOLDER_PATH = "/storage/brno2/home/xkaska01/master/my_implementation/results/stats_with_defense_second"
FOLDER_PATH = "/storage/brno2/home/xkaska01/master/my_implementation/results/stats_without_defense"


def build_heatmaps(folder_path: str):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Ve složce {folder_path} jsem nenašel žádné .csv soubory.")

    all_rows = []

    for file_path in csv_files:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Načítám {file_path} jako model '{model_name}'")

        df = pd.read_csv(file_path)

        score_cols = [c for c in df.columns if c.startswith("score_")]
        if not score_cols:
            raise ValueError(f"V souboru {file_path} jsem nenašel žádné sloupce score_*")

        if "attack_type" not in df.columns:
            raise ValueError(f"V souboru {file_path} chybí sloupec 'attack_type'")

        # statistiky přes score_* v rámci každého řádku
        df["mean_score"] = df[score_cols].mean(axis=1)
        df["std_score"] = df[score_cols].std(axis=1, ddof=0)

        # pokud je více řádků pro stejný attack_type, zprůměrujeme je
        df_attack = (
            df.groupby("attack_type")[["mean_score", "std_score"]]
            .mean()
            .reset_index()
        )

        df_attack["model"] = model_name
        all_rows.append(df_attack)

    combined = pd.concat(all_rows, ignore_index=True)

    heatmap_mean = combined.pivot(index="model", columns="attack_type", values="mean_score")
    heatmap_std = combined.pivot(index="model", columns="attack_type", values="std_score")

    return heatmap_mean, heatmap_std


def align_heatmaps(A: pd.DataFrame, B: pd.DataFrame):
    all_models = A.index.union(B.index)
    all_attacks = A.columns.union(B.columns)

    A = A.reindex(index=all_models, columns=all_attacks)
    B = B.reindex(index=all_models, columns=all_attacks)

    return A, B


def plot_side_heatmap(
    diff_df: pd.DataFrame,
    out_png: str,
    title_right: str,
    cbar_label: str,
    vmin: float = 0,
    vmax: float = 10,
    cmap: str = "Greens"
):
    # řazení ještě před transpozicí
    model_order = diff_df.mean(axis=1).sort_values(ascending=False).index
    attack_order = diff_df.mean(axis=0).sort_values(ascending=False).index
    diff_df = diff_df.loc[model_order, attack_order]

    # "na bok"
    diff_df = diff_df.T
    diff_masked = diff_df.where(diff_df > 0)

    print(f"Tvar výsledné matice pro {out_png}: {diff_df.shape}")

    fig, ax = plt.subplots(
        figsize=(max(12, 0.45 * diff_df.shape[1]), max(8, 0.35 * diff_df.shape[0]))
    )

    im = ax.imshow(
        diff_masked.values,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )

    # osa X = modely
    ax.set_xticks(np.arange(diff_masked.shape[1]))
    ax.set_xticklabels(diff_masked.columns, rotation=90, fontsize=8)

    # osa Y = attack types
    ax.set_yticks(np.arange(diff_masked.shape[0]))
    ax.set_yticklabels(diff_masked.index, fontsize=10)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_xlabel("Model")
    ax.set_ylabel("Attack type")

    fig.text(
        0.995, 0.5,
        title_right,
        rotation=-90,
        va="center",
        ha="right",
        fontsize=11
    )

    # hodnoty do buněk
    for i in range(diff_masked.shape[0]):
        for j in range(diff_masked.shape[1]):
            val = diff_masked.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08)
    cbar.set_label(cbar_label)

    plt.tight_layout(rect=[0, 0, 0.98, 1])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] PNG uloženo do: {out_png}")


# 1) načtení mean a std pro oba foldery
heatmap_mean_A, heatmap_std_A = build_heatmaps(FOLDER_PATH)
heatmap_mean_B, heatmap_std_B = build_heatmaps(SECOND_FOLDER_PATH)

# pokud ten +1 opravdu chceš zachovat jen pro mean:
heatmap_mean_A = heatmap_mean_A + 2

# 2) zarovnání
heatmap_mean_A, heatmap_mean_B = align_heatmaps(heatmap_mean_A, heatmap_mean_B)
heatmap_std_A, heatmap_std_B = align_heatmaps(heatmap_std_A, heatmap_std_B)

# 3) rozdíly
diff_mean = heatmap_mean_A - heatmap_mean_B
diff_std = heatmap_std_A - heatmap_std_B

# 4) heatmapa rozdílu průměrů
plot_side_heatmap(
    diff_df=diff_mean,
    out_png="heatmap_diff_mean_A_minus_B.png",
    title_right="Mean improvement across models and attack types",
    cbar_label="Mean score difference",
    vmin=0,
    vmax=10,
    cmap="Greens"
)

# 5) heatmapa rozdílu směrodatné odchylky
plot_side_heatmap(
    diff_df=diff_std,
    out_png="heatmap_diff_std_A_minus_B.png",
    title_right="Standard deviation change across models and attack types",
    cbar_label="Std deviation difference",
    vmin=0,
    vmax=max(1, np.nanmax(diff_std.values)),
    cmap="Oranges"
)

# 6) TXT výstup
TXT_OUT = "heatmap_stats_A_minus_B.txt"
with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("=== NUMERICKÉ HEATMAPY ROZDÍLŮ ===\n")
    f.write(f"A = {FOLDER_PATH}\n")
    f.write(f"B = {SECOND_FOLDER_PATH}\n\n")

    f.write("=== ROZDÍL PRŮMĚRŮ (A - B) ===\n\n")
    f.write(diff_mean.round(3).to_string())
    f.write("\n\n")

    f.write("=== ROZDÍL SMĚRODATNÝCH ODCHYLEK (A - B) ===\n\n")
    f.write(diff_std.round(3).to_string())
    f.write("\n\n")

    f.write("=== PRŮMĚR PŘES MODELY (attack_type) - MEAN DIFF ===\n\n")
    f.write(diff_mean.mean(axis=0).sort_values(ascending=False).round(3).to_string())
    f.write("\n\n")

    f.write("=== PRŮMĚR PŘES ÚTOKY (model) - MEAN DIFF ===\n\n")
    f.write(diff_mean.mean(axis=1).sort_values(ascending=False).round(3).to_string())
    f.write("\n\n")

    f.write("=== PRŮMĚR PŘES MODELY (attack_type) - STD DIFF ===\n\n")
    f.write(diff_std.mean(axis=0).sort_values(ascending=False).round(3).to_string())
    f.write("\n\n")

    f.write("=== PRŮMĚR PŘES ÚTOKY (model) - STD DIFF ===\n\n")
    f.write(diff_std.mean(axis=1).sort_values(ascending=False).round(3).to_string())
    f.write("\n\n")

    f.write("=== A: MEAN ± STD pro jednotlivé modely a attacky ===\n\n")
    for model in heatmap_mean_A.index:
        f.write(f"[{model}]\n")
        for attack in heatmap_mean_A.columns:
            mean_val = heatmap_mean_A.loc[model, attack]
            std_val = heatmap_std_A.loc[model, attack]
            if pd.notna(mean_val):
                if pd.notna(std_val):
                    f.write(f"  {attack}: {mean_val:.3f} ± {std_val:.3f}\n")
                else:
                    f.write(f"  {attack}: {mean_val:.3f} ± NaN\n")
        f.write("\n")

    f.write("=== B: MEAN ± STD pro jednotlivé modely a attacky ===\n\n")
    for model in heatmap_mean_B.index:
        f.write(f"[{model}]\n")
        for attack in heatmap_mean_B.columns:
            mean_val = heatmap_mean_B.loc[model, attack]
            std_val = heatmap_std_B.loc[model, attack]
            if pd.notna(mean_val):
                if pd.notna(std_val):
                    f.write(f"  {attack}: {mean_val:.3f} ± {std_val:.3f}\n")
                else:
                    f.write(f"  {attack}: {mean_val:.3f} ± NaN\n")
        f.write("\n")

print(f"[OK] TXT uloženo do: {TXT_OUT}")

overall_mean_improvement = np.nanmean(diff_mean.values)
overall_std_change = np.nanmean(diff_std.values)

print("\n==============================")
print(f"GLOBAL MEAN IMPROVEMENT: {overall_mean_improvement:.3f}")
print(f"GLOBAL STD CHANGE:       {overall_std_change:.3f}")
print("==============================\n")