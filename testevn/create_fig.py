import sys
import re
import pandas as pd
import matplotlib.pyplot as plt


def find_judge_column(df: pd.DataFrame) -> str:
    candidates = [col for col in df.columns if "judge" in col.lower()]
    if not candidates:
        raise ValueError(
            f"Nenašel jsem žádný sloupec obsahující 'judge'. Dostupné sloupce: {list(df.columns)}"
        )
    if len(candidates) > 1:
        print(f"Nalezeno více judge sloupců, používám první: {candidates[0]}")
    return candidates[0]


def find_human_column(df: pd.DataFrame) -> str:
    candidates = [col for col in df.columns if col.lower() == "human_score"]
    if not candidates:
        # fallback: zkus najít něco, co human_score obsahuje
        candidates = [col for col in df.columns if "human_score" in col.lower()]

    if not candidates:
        raise ValueError(
            f"Nenašel jsem sloupec 'human_score'. Dostupné sloupce: {list(df.columns)}"
        )

    if len(candidates) > 1:
        print(f"Nalezeno více human_score sloupců, používám první: {candidates[0]}")

    return candidates[0]


def parse_score(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if re.fullmatch(r"(10|[0-9])", text):
        return int(text)

    match = re.search(r"\b(10|[0-9])\b", text)
    if match:
        return int(match.group(1))

    return None


def prepare_counts(series: pd.Series) -> pd.Series:
    scores = series.apply(parse_score).dropna().astype(int)
    scores = scores[(scores >= 0) & (scores <= 10)]
    return scores.value_counts().reindex(range(11), fill_value=0)


def main():
    if len(sys.argv) < 2:
        print("Použití: python plot_scores_compare.py soubor.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    judge_col = find_judge_column(df)
    human_col = find_human_column(df)

    print(f"Používám judge sloupec: {judge_col}")
    print(f"Používám human sloupec: {human_col}")

    judge_counts = prepare_counts(df[judge_col])
    human_counts = prepare_counts(df[human_col])

    if judge_counts.sum() == 0:
        raise ValueError(f"Ve sloupci '{judge_col}' se nepodařilo najít žádná validní skóre 0-10.")

    if human_counts.sum() == 0:
        raise ValueError(f"Ve sloupci '{human_col}' se nepodařilo najít žádná validní skóre 0-10.")

    x = list(range(11))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], judge_counts.values, width=width, label=judge_col)
    plt.bar([i + width / 2 for i in x], human_counts.values, width=width, label=human_col)

    plt.xticks(x)
    plt.xlabel("Skóre")
    plt.ylabel("Četnost")
    plt.title("Porovnání rozdělení skóre: Judge vs Human")
    plt.legend()
    plt.tight_layout()

    output_path = sys.argv[1] + ".png"
    plt.savefig(output_path, dpi=200)
    print(f"Graf uložen do: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()