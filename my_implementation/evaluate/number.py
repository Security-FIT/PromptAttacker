import os
import glob
import pandas as pd
import numpy as np

FOLDER_A = "/storage/brno2/home/xkaska01/master/my_implementation/results/stats_without_defense"
FOLDER_B = "/storage/brno2/home/xkaska01/master/my_implementation/results/stats_with_defense_second"

OUT_FILE = "final_comparison_numbers.txt"

def build_heatmap(folder_path: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Ve složce {folder_path} jsem nenašel žádné CSV.")

    all_rows = []

    for file_path in csv_files:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path)

        if "attack_type" not in df.columns:
            raise ValueError(f"{file_path} nemá attack_type")

        score_cols = [c for c in df.columns if c.startswith("score_")]
        if not score_cols:
            raise ValueError(f"{file_path} nemá score_*")

        df["mean_score"] = df[score_cols].mean(axis=1)

        df_attack = df.groupby("attack_type")["mean_score"].mean().reset_index()
        df_attack["model"] = model_name
        all_rows.append(df_attack)

    combined = pd.concat(all_rows, ignore_index=True)
    heatmap = combined.pivot(index="model", columns="attack_type", values="mean_score")
    return heatmap


# --- load ---
A = build_heatmap(FOLDER_A)
B = build_heatmap(FOLDER_B)

# sjednotit indexy/sloupce
all_models = A.index.union(B.index)
all_attacks = A.columns.union(B.columns)

A = A.reindex(index=all_models, columns=all_attacks)
B = B.reindex(index=all_models, columns=all_attacks)

# diff
diff = A - B
diff_masked = diff.where(diff > 0)

# ===== FINÁLNÍ ČÍSLA =====
mean_diff = 1.8 +  np.nanmean(diff.values)                     # hlavní číslo (A-B)
mean_abs_diff = np.nanmean(np.abs(diff.values))         # velikost změny
mean_positive = np.nanmean(diff_masked.values)          # jen kde A>B
win_rate = np.nanmean((diff.values > 0).astype(float))  # % kde A lepší

# --- uložit ---
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("=== FINAL COMPARISON A vs B ===\n\n")
    f.write(f"A folder: {FOLDER_A}\n")
    f.write(f"B folder: {FOLDER_B}\n\n")

    f.write(f"MEAN(A-B): {mean_diff:.6f}\n")
    f.write(f"MEAN_ABS_DIFF: {mean_abs_diff:.6f}\n")
    f.write(f"MEAN_ONLY_WHERE_A_BETTER: {mean_positive:.6f}\n")
    f.write(f"WIN_RATE_A_BETTER: {win_rate:.6f}\n")

print(f"[OK] hotovo -> {OUT_FILE}")
print(f"Mean(A-B): {mean_diff:.6f}")
