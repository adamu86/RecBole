import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
from tqdm import tqdm

def analyze_sessions(file_path, output_dir):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Analyzing {file_path} ...\n")
    
    session_lengths = []
    session_durations = []
    item_counts = Counter()
    session_dates = []
    
    total_interactions = 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading data"):
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                
                try:
                    timestamp = int(parts[1])
                    tracks = json.loads(parts[3])
                    
                    if not tracks:
                        continue
                        
                    # Długość w liczbie utworów
                    length = len(tracks)
                    session_lengths.append(length)
                    
                    # Czas trwania
                    ps_values = [t["ps"] for t in tracks]
                    duration = max(ps_values) - min(ps_values)
                    session_durations.append(duration)
                    
                    # Popularność utworów
                    for t in tracks:
                        item_counts[t["id"]] += 1
                        total_interactions += 1
                        
                    # Wyciągnięcie daty z timestampu (rok-miesiąc-dzień)
                    date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                    session_dates.append(date_str)
                    
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not session_lengths:
        print("No valid data found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Ustawienie ładniejszego stylu
    sns.set_theme(style="whitegrid")
    
    # ---------------------------------------------------------
    # 1. Rozkład długości sesji (liczba utworów)
    # ---------------------------------------------------------
    print("Generating session length plot...")
    plt.figure(figsize=(10, 6))
    
    max_len = max(session_lengths)
    # Ucinamy oś X przy ok. 100 jeśli są jakieś duże outliery, chociaż po Twoim filtrze max to 100
    sns.histplot(session_lengths, bins=range(min(session_lengths), min(max_len + 2, 105)), kde=False, color='skyblue')
    
    plt.title('Rozkład długości sesji w zbiorze (liczba utworów)', fontsize=14, pad=15)
    plt.xlabel('Liczba utworów w sesji', fontsize=12)
    plt.ylabel('Liczba sesji', fontsize=12)
    plt.xlim(0, min(100, max_len))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_session_lengths.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. Rozkład popularności utworów (Long Tail)
    # ---------------------------------------------------------
    print("Generating item popularity (Long Tail) plot...")
    counts = sorted(list(item_counts.values()), reverse=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(counts)), counts, color='crimson', linewidth=2)
    plt.fill_between(range(len(counts)), counts, alpha=0.2, color='crimson')
    
    plt.title('Rozkład popularności utworów (Long Tail)', fontsize=14, pad=15)
    plt.xlabel('Indeks utworu (od najpopularniejszego)', fontsize=12)
    plt.ylabel('Liczba odtworzeń (skala logarytmiczna)', fontsize=12)
    plt.yscale('log') # Skala logarytmiczna jest idealna dla long tail
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_item_popularity_long_tail.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. Liczba sesji w poszczególnych dniach
    # ---------------------------------------------------------
    print("Generating sessions per day plot...")
    date_counts = pd.Series(session_dates).value_counts().sort_index()
    date_counts.index = pd.to_datetime(date_counts.index)
    
    plt.figure(figsize=(14, 6))
    plt.plot(date_counts.index, date_counts.values, marker='.', linestyle='-', color='teal', linewidth=1.5)
    
    plt.title('Liczba sesji w poszczególnych dniach w zbiorze danych', fontsize=14, pad=15)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Liczba unikalnych sesji', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_sessions_per_day.png'), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 4. BONUS: Rozkład czasu trwania sesji (w minutach)
    # ---------------------------------------------------------
    print("Generating session duration plot...")
    durations_min = [d / 60 for d in session_durations if d <= 7200] # Obcinamy wizualizację do 2 godzin
    
    plt.figure(figsize=(10, 6))
    sns.histplot(durations_min, bins=60, color='purple', kde=True, alpha=0.6)
    
    plt.title('Rozkład czasu trwania sesji (ograniczony do pierwszych 2 godzin)', fontsize=14, pad=15)
    plt.xlabel('Czas trwania (minuty)', fontsize=12)
    plt.ylabel('Liczba sesji', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_session_durations.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 5. Kubełkowe statystyki popularności (Bucket Statistics)
    # ---------------------------------------------------------
    print("Generating popularity bucket statistics...")
    bins = [0, 24, 50, 100, 500, 1000, float('inf')] # Dodano 0-24
    labels = ['<25', '25-50', '51-100', '101-500', '501-1000', '1000+']
    
    counts_series = pd.Series(list(item_counts.values()))
    
    # Create buckets
    buckets = pd.cut(counts_series, bins=bins, labels=labels, right=True)
    bucket_counts = buckets.value_counts().sort_index()
    bucket_interactions = counts_series.groupby(buckets, observed=False).sum()
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=bucket_counts.index, y=bucket_counts.values, hue=bucket_counts.index, palette="viridis", legend=False)
    plt.title('Rozkład utworów w kubełkach popularności (Liczba odtworzeń)', fontsize=14, pad=15)
    plt.xlabel('Przedziały liczby odtworzeń (Kubełki)', fontsize=12)
    plt.ylabel('Liczba unikalnych utworów', fontsize=12)
    
    # Add labels on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{int(p.get_height()):,}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_item_popularity_buckets.png'), dpi=300)
    plt.close()

    print(f"\nAnaliza zakończona! Wykresy zostały zapisane w folderze: {os.path.abspath(output_dir)}")
    
    # Print Bucket Stats to console
    print("\n" + "=" * 80)
    print("--- STATYSTYKI KUBEŁKOWE (Popularność utworów) ---")
    print(f"{'Kubełek (odtworzenia)':<25} | {'Liczba utworów':<15} | {'Suma interakcji':<15} | {'% wszystkich interakcji'}")
    print("-" * 80)
    for label in labels:
        n_items = bucket_counts[label]
        n_inter = bucket_interactions[label]
        pct = (n_inter / total_interactions) * 100 if total_interactions > 0 else 0
        print(f"{label:<25} | {n_items:<15,} | {int(n_inter):<15,} | {pct:.2f}%")
    print("=" * 80)
    print("-" * 50)
    print(f"PODSUMOWANIE STATYSTYK ZBIORU:")
    print(f"  Liczba sesji ogółem:      {len(session_lengths):,}")
    print(f"  Liczba interakcji ogółem: {total_interactions:,}")
    print(f"  Unikalnych utworów:       {len(item_counts):,}")
    print(f"  Średnia długość sesji:    {sum(session_lengths)/len(session_lengths):.2f} utworów")
    print(f"  Mediana długości sesji:   {pd.Series(session_lengths).median():.1f} utworów")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analiza zbioru sessions.tsv po przetworzeniu")
    parser.add_argument("--file", type=str, default="dataset_raw/sessions.tsv", help="Ścieżka do sessions.tsv")
    parser.add_argument("--out", type=str, default="analysis_plots", help="Katalog, do którego zapisać wykresy")
    args = parser.parse_args()
    
    analyze_sessions(args.file, args.out)
