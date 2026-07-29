import os
import pandas as pd
from collections import Counter

def analyze_dataset_directory(dataset_dir):
    dir_name = os.path.basename(dataset_dir)
    print(f"\n" + "=" * 80)
    print(f" ANALIZA ZBIORU: {dir_name}")
    print("=" * 80)

    # Używamy os.listdir zamiast glob (omija problem z nawiasami kwadratowymi '[' i ']' w glob)
    all_files = os.listdir(dataset_dir)

    # 1. Analiza oryginalnych długości sesji z pliku .inter
    inter_files = [
        f for f in all_files 
        if f.endswith('.inter') and not f.endswith(('.train.inter', '.valid.inter', '.test.inter'))
    ]
    
    if inter_files:
        inter_path = os.path.join(dataset_dir, inter_files[0])
        print(f"\n--- 1. Oryginalne sesje (przed augmentacją / prefiksowaniem): {os.path.basename(inter_path)} ---")
        
        session_counts = Counter()
        with open(inter_path, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
            # Sprawdź separator (tabulator lub spacja)
            sep = "\t" if "\t" in header_line else " "
            header = header_line.split(sep)
            
            session_idx = -1
            for idx, col in enumerate(header):
                if col.startswith("session_id"):
                    session_idx = idx
                    break
            
            if session_idx != -1:
                for line in f:
                    parts = line.strip().split(sep)
                    if len(parts) > session_idx:
                        session_counts[parts[session_idx]] += 1

        lengths = list(session_counts.values())
        total_sessions = len(lengths)
        if total_sessions > 0:
            len_series = pd.Series(lengths)
            print(f"Łączna liczba oryginalnych sesji: {total_sessions:,}")
            print(f"Średnia długość sesji: {len_series.mean():.2f}")
            print(f"Mediana długości sesji: {len_series.median():.0f}")
            print(f"Min długość: {len_series.min()}, Max długość: {len_series.max()}")
            
            c2 = sum(1 for l in lengths if l == 2)
            c3 = sum(1 for l in lengths if l == 3)
            c2_3 = c2 + c3
            c4_5 = sum(1 for l in lengths if 4 <= l <= 5)
            c6_10 = sum(1 for l in lengths if 6 <= l <= 10)
            c_gt10 = sum(1 for l in lengths if l > 10)

            print("\nRozkład oryginalnych długości sesji:")
            print(f"  * Długość == 2:      {c2:7,} sesji ({c2/total_sessions*100:6.2f}%)")
            print(f"  * Długość == 3:      {c3:7,} sesji ({c3/total_sessions*100:6.2f}%)")
            print(f"  * Długość 2-3 (<=3): {c2_3:7,} sesji ({c2_3/total_sessions*100:6.2f}%)  <-- 🚨 KRÓTKIE SESJE")
            print(f"  * Długość 4-5:       {c4_5:7,} sesji ({c4_5/total_sessions*100:6.2f}%)")
            print(f"  * Długość 6-10:      {c6_10:7,} sesji ({c6_10/total_sessions*100:6.2f}%)")
            print(f"  * Długość > 10:      {c_gt10:7,} sesji ({c_gt10/total_sessions*100:6.2f}%)")

    # 2. Analiza prefiksów w pliku .train.inter
    train_files = [f for f in all_files if f.endswith('.train.inter')]
    if train_files:
        train_path = os.path.join(dataset_dir, train_files[0])
        print(f"\n--- 2. Przykłady treningowe (długość prefiksu item_id_list): {os.path.basename(train_path)} ---")
        
        prefix_lengths = []
        with open(train_path, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
            sep = "\t" if "\t" in header_line else " "
            header = header_line.split(sep)
            
            seq_idx = -1
            for idx, col in enumerate(header):
                if col.startswith("item_id_list"):
                    seq_idx = idx
                    break
            
            if seq_idx != -1:
                for line in f:
                    parts = line.strip().split(sep)
                    if len(parts) > seq_idx:
                        seq_str = parts[seq_idx].strip()
                        p_len = len(seq_str.split()) if seq_str else 0
                        prefix_lengths.append(p_len)

        total_train = len(prefix_lengths)
        if total_train > 0:
            p_series = pd.Series(prefix_lengths)
            print(f"Łączna liczba próbek treningowych: {total_train:,}")
            print(f"Średnia długość prefiksu: {p_series.mean():.2f}")
            print(f"Mediana długości prefiksu: {p_series.median():.0f}")
            
            p1 = sum(1 for l in prefix_lengths if l == 1)
            p2 = sum(1 for l in prefix_lengths if l == 2)
            p1_2 = p1 + p2
            p3 = sum(1 for l in prefix_lengths if l == 3)
            p1_3 = p1_2 + p3
            p4_5 = sum(1 for l in prefix_lengths if 4 <= l <= 5)
            p6_10 = sum(1 for l in prefix_lengths if 6 <= l <= 10)
            p_gt10 = sum(1 for l in prefix_lengths if l > 10)

            print("\nRozkład długości prefiksów (item_id_list) w danych treningowych:")
            print(f"  * Prefiks == 1:       {p1:7,} próbek ({p1/total_train*100:6.2f}%)")
            print(f"  * Prefiks == 2:       {p2:7,} próbek ({p2/total_train*100:6.2f}%)")
            print(f"  * Prefiks 1-2 (<=2):  {p1_2:7,} próbek ({p1_2/total_train*100:6.2f}%)  <-- 🚨 PREFIKS ≤ 2")
            print(f"  * Prefiks 1-3 (<=3):  {p1_3:7,} próbek ({p1_3/total_train*100:6.2f}%)")
            print(f"  * Prefiks 4-5:        {p4_5:7,} próbek ({p4_5/total_train*100:6.2f}%)")
            print(f"  * Prefiks 6-10:       {p6_10:7,} próbek ({p6_10/total_train*100:6.2f}%)")
            print(f"  * Prefiks > 10:       {p_gt10:7,} próbek ({p_gt10/total_train*100:6.2f}%)")

def main():
    dataset_base = "dataset"
    if not os.path.exists(dataset_base):
        print(f"Katalog {dataset_base} nie istnieje.")
        return

    subdirs = [os.path.join(dataset_base, d) for d in os.listdir(dataset_base) 
               if os.path.isdir(os.path.join(dataset_base, d))]
    
    for d in sorted(subdirs):
        analyze_dataset_directory(d)

if __name__ == "__main__":
    main()
