import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FOLDER_PATH = "/storage/brno2/home/xkaska01/master/my_implementation/results/legitimate"
OUT_PNG = "legitimate_boxplot.png"

csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
if not csv_files:
    raise FileNotFoundError("Žádné CSV")

all_data = []

for file_path in csv_files:
    model_name = os.path.splitext(os.path.basename(file_path))[0]
    print("[INFO]", model_name)

    df = pd.read_csv(file_path)

    score_cols = [c for c in df.columns if c.startswith("score_")]
    if not score_cols:
        raise ValueError(f"{file_path}: missing score cols")

    # 🔥 vezmeme všechny hodnoty napřímo
    for col in score_cols:
        for val in df[col].dropna():
            all_data.append({
                "model": model_name,
                "score": val
            })

plot_df = pd.DataFrame(all_data)

# seřadit podle mediánu
order = (
    plot_df.groupby("model")["score"]
    .median()
    .sort_values(ascending=False)
    .index
)

plt.figure(figsize=(10, max(6, 0.35 * len(order))))

sns.boxplot(
    data=plot_df,
    y="model",
    x="score",
    order=order,
    orient="h",
    # color="blue",
    showfliers=False
)

plt.xlim(0, 10)
plt.xlabel("Porozumění neškodnému promptu (0–10)")
plt.ylabel("Model")
plt.title("Distribuce hodnocení porozumění neškodnému promptu pro jednotlivé modely")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()

print("[OK] hotovo:", OUT_PNG)
