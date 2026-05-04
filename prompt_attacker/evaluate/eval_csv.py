import csv

# ZDE případně přepiš název souboru:
CSV_FILE = '/storage/brno2/home/xkaska01/master/prompt_attacker/results/stats_without_defense/internlm2.5:latest.csv'

def main():
    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Vezmeme jen číselné sloupce (ignorujeme attack_type)
            values = []
            for key, value in row.items():
                if key == "attack_type":
                    continue
                try:
                    num = float(value)
                    values.append(num)
                except ValueError:
                    pass

            if values:
                avg = sum(values) / len(values)
                print(f"{row['attack_type']}: {avg:.4f}")
            else:
                print(f"{row['attack_type']}: NO NUMERIC DATA")

if __name__ == "__main__":
    main()
