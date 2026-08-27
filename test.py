import os
import glob
import pandas as pd

BACKUP_DIR = "dataset/"
PREFIXES = ["30music", "lastfm1k"]

def compute_stats(alias, base_dir):
    train_path = os.path.join(base_dir, f"{alias}.train.inter")
    test_path = os.path.join(base_dir, f"{alias}.test.inter")

    if not (os.path.isfile(train_path) and os.path.isfile(test_path)):
        return None

    train = pd.read_csv(train_path, sep="\t", low_memory=False)
    test = pd.read_csv(test_path, sep="\t", low_memory=False)

    train_items = set(train["item_id:token"])
    test_items = set(test["item_id:token"])

    cold_start_items = test_items - train_items

    item_pct = (len(cold_start_items) / len(test_items) * 100) if test_items else 0.0

    cold_start_interactions = test["item_id:token"].isin(cold_start_items).sum()
    inter_pct = (cold_start_interactions / len(test) * 100) if len(test) else 0.0

    return {
        "alias": alias,
        "train_items": len(train_items),
        "test_items": len(test_items),
        "cold_start_items": len(cold_start_items),
        "cold_start_items_pct": item_pct,
        "test_interactions": len(test),
        "cold_start_interactions": cold_start_interactions,
        "cold_start_interactions_pct": inter_pct,
    }

def main():
    results = []

    for prefix in PREFIXES:
        pattern = os.path.join(BACKUP_DIR, f"{prefix}__*")
        for dir_path in sorted(glob.glob(pattern)):
            if not os.path.isdir(dir_path):
                continue
            alias = os.path.basename(dir_path)
            stats = compute_stats(alias, dir_path)
            if stats is None:
                print(f"Pominięto {alias} (brak .train.inter lub .test.inter)")
                continue
            results.append(stats)

            print(f"\n--- {alias} ---")
            print(f"Unikalne itemy w train: {stats['train_items']:,}")
            print(f"Unikalne itemy w test:  {stats['test_items']:,}")
            print(f"Itemy w test, nie w train: {stats['cold_start_items']:,} "
                  f"({stats['cold_start_items_pct']:.4f}%)")
            print(f"Interakcje testowe z takimi itemami: {stats['cold_start_interactions']:,} "
                  f"({stats['cold_start_interactions_pct']:.4f}%)")

    if results:
        df = pd.DataFrame(results)
        out_path = "cold_start_stats.csv"
        df.to_csv(out_path, index=False)
        print(f"\nZapisano zbiorczą tabelę do {out_path}")
        print(df.to_string(index=False))
    else:
        print("\nNie znaleziono żadnych pasujących katalogów.")

if __name__ == "__main__":
    main()