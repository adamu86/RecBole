import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

DATA_PATH_RAW = "dataset_raw/"
DATA_FILE = "sessions"
OUTPUT_DIR = "plots"
ANALYSIS_DIR = "analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def get_line_count(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_sessions_data():
    """
    Ładuje dane z surowego pliku i zwraca listę ID sesji, ich czasów trwania oraz timestampy.
    """
    input_path = os.path.join(DATA_PATH_RAW, f"{DATA_FILE}.idomaar")
    total_lines = get_line_count(input_path)

    ids = []
    playtimes = []
    timestamps = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Wczytywanie sesji"):
            line = line.strip()
            if not line:
                continue
            try:
                after_prefix = line[len("event.session\t"):]
                parts = after_prefix.split("\t", 2)
                session_id = parts[0]
                timestamp = int(parts[1])
                
                stats_json = parts[2][:parts[2].find("} {") + 1]
                stats = json.loads(stats_json)

                pt = stats["playtime"]
                if pt != -1:
                    ids.append(session_id)
                    playtimes.append(pt)
                    timestamps.append(timestamp)
            except (json.JSONDecodeError, ValueError, IndexError, KeyError):
                continue

    return np.array(ids), np.array(playtimes), np.array(timestamps)


def print_stats(playtimes):
    playtimes_min = playtimes / 60.0
    print(f"\n{'='*40}")
    print(f" STATYSTYKI PLAYTIME ")
    print(f"{'='*40}")
    print(f"Liczba sesji:  {len(playtimes):,}")
    print(f"Średnia:       {playtimes.mean():.1f} s ({playtimes_min.mean():.1f} min)")
    print(f"Mediana:       {np.median(playtimes):.1f} s ({np.median(playtimes_min):.1f} min)")
    print(f"Max:           {playtimes.max()/3600:.2f} h")
    print(f"P95:           {np.percentile(playtimes, 95):.0f} s")
    print(f"{'='*40}\n")


def plot_seconds_distribution(playtimes, limit=200):
    """Krótki czas (sekundy)"""
    filtered = playtimes[(playtimes >= 0) & (playtimes <= limit)]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bins = np.arange(0, limit + 2)
    ax.hist(filtered, bins=bins, color="#4A90D9", edgecolor="white", linewidth=0.2)
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Sesje")
    ax.set_title(f"Rozkład do {limit}s")
    
    path = os.path.join(OUTPUT_DIR, "dist_seconds.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Zapisano: {path}")


def plot_hours_distribution(playtimes):
    """Pełny rozkład (godziny)"""
    hours = playtimes / 3600.0
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(hours, bins=100, color="#D94A4A", log=True)
    ax.set_xlabel("Czas [h]")
    ax.set_ylabel("Sesje (log)")
    ax.set_title("Rozkład czasu (godziny)")
    
    path = os.path.join(OUTPUT_DIR, "dist_hours.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Zapisano: {path}")


def analyze_outliers(session_ids, playtimes):
    """Outlierzy: boxplot, ogon i zapisanie ID do pliku"""
    hours = playtimes / 3600.0
    
    # 1. Boxplot
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.boxplot(hours, vert=False, patch_artist=True, boxprops=dict(facecolor="#4AD94A"))
    ax.set_xlabel("Czas [h]")
    ax.set_title("Outlierzy (Boxplot)")
    path_box = os.path.join(OUTPUT_DIR, "outliers_boxplot.png")
    plt.savefig(path_box, dpi=120)
    plt.close()

    # 2. Rozkład ogona (P99+)
    p99 = np.percentile(hours, 99)
    tail = hours[hours > p99]
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.hist(tail, bins=50, color="#D9A44A")
    ax2.set_title(f"Ogon rozkładu (>P99: {p99:.1f}h)")
    path_tail = os.path.join(OUTPUT_DIR, "outliers_tail.png")
    plt.savefig(path_tail, dpi=120)
    plt.close()

    # 3. Zapis ID outlierów (metoda IQR)
    q1, q3 = np.percentile(playtimes, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    
    outlier_mask = playtimes > upper_bound
    outlier_ids = session_ids[outlier_mask]
    outlier_vals = playtimes[outlier_mask]
    
    save_path = os.path.join(ANALYSIS_DIR, "outlier_sessions.txt")
    with open(save_path, "w") as f:
        f.write(f"ID\tPlaytime[s]\tPlaytime[h]\n")
        for sid, val in zip(outlier_ids, outlier_vals):
            f.write(f"{sid}\t{val}\t{val/3600:.2f}\n")
            
    print(f"Zapisano {len(outlier_ids):,} outlierów (> {upper_bound/3600:.2f}h) do: {save_path}")
    print(f"Zapisano wykresy: {path_box}, {path_tail}")


def plot_sessions_per_day(timestamps):
    """
    Zlicza sesje na dzień i pokazuje rozkład przez 365 dni.
    """
    # Konwersja do datetime
    dates = pd.to_datetime(timestamps, unit='s').date
    df = pd.DataFrame({'date': dates})
    
    # Zliczanie sesji na dzień
    counts = df.groupby('date').size().reset_index(name='count')
    counts = counts.sort_values('date')
    
    # Ograniczenie do ostatnich 365 dni z danych (lub całości jeśli mniej)
    if len(counts) > 365:
        counts = counts.tail(365)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(counts['date'], counts['count'], color="#7B4AD9", linewidth=2)
    ax.fill_between(counts['date'], counts['count'], color="#7B4AD9", alpha=0.2)
    
    ax.set_xlabel("Data")
    ax.set_ylabel("Liczba sesji")
    ax.set_title(f"Rozkład liczby sesji na dzień (ostatnie {len(counts)} dni)")
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, "sessions_per_day.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Zapisano: {path}")


if __name__ == "__main__":
    # 1. Załaduj raz
    s_ids, s_pts, s_ts = load_sessions_data()
    
    # # 2. Statystyki w konsoli
    # print_stats(s_pts)
    
    # # 3. Poszczególne zadania
    # plot_seconds_distribution(s_pts, limit=200)
    # plot_hours_distribution(s_pts)
    # analyze_outliers(s_ids, s_pts)
    plot_sessions_per_day(s_ts)
    
    print("\nAnaliza zakończona.")
