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


def analyze_tsv_time_window(days_from=365, days_to=65):
    """
    Zlicza medianę i percentyle dla sesji z pliku dataset_raw/sessions.tsv
    z określonego przedziału dni (względem maksymalnego timestampu).
    """
    input_path = os.path.join(DATA_PATH_RAW, "sessions.tsv")
    if not os.path.exists(input_path):
        print(f"Nie znaleziono pliku: {input_path}")
        return

    print(f"\nZnajdowanie maksymalnego timestampu w {input_path}...")
    max_timestamp = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    ts = int(parts[1])
                    if ts > max_timestamp:
                        max_timestamp = ts
                except ValueError:
                    pass

    lower_bound = max_timestamp - days_from * 86400
    upper_bound = max_timestamp - days_to * 86400

    lengths = []
    playtimes = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Filtrowanie i analiza ({days_from}-{days_to} dni)"):
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                try:
                    ts = int(parts[1])
                    if lower_bound <= ts <= upper_bound:
                        tracks = json.loads(parts[3])
                        lengths.append(len(tracks))
                        
                        # Czas trwania to suma 'pt' z poszczególnych tracków
                        pt_sum = sum(t.get("pt", 0) for t in tracks)
                        playtimes.append(pt_sum)
                except (ValueError, json.JSONDecodeError):
                    continue

    if not lengths:
        print(f"\nBrak sesji w pliku {input_path} dla przedziału {days_from}-{days_to} dni.")
        return

    lengths = np.array(lengths)
    playtimes = np.array(playtimes)

    print(f"\n{'='*50}")
    print(f" STATYSTYKI {input_path} ({days_from}-{days_to} DNI) ")
    print(f"{'='*50}")
    print(f"Liczba sesji:  {len(lengths):,}")
    
    print(f"\n--- DŁUGOŚĆ SESJI (LICZBA UTWORÓW) ---")
    print(f"Średnia:       {lengths.mean():.1f}")
    print(f"Mediana:       {np.median(lengths):.1f}")
    for p in [25, 75, 90, 95, 99]:
        print(f"P{p}:           {np.percentile(lengths, p):.0f}")

    print(f"\n--- CZAS TRWANIA (PLAYTIME Z UTWORÓW) ---")
    print(f"Średnia:       {playtimes.mean():.1f} s")
    print(f"Mediana:       {np.median(playtimes):.1f} s")
    for p in [25, 75, 90, 95, 99]:
        print(f"P{p}:           {np.percentile(playtimes, p):.0f} s")
    print(f"{'='*50}\n")


def session_length_percentiles():
    """
    Wylicza percentyle długości sesji (liczba tracków) z pliku dataset_raw/sessions.tsv.
    Brak filtrowania po oknie czasowym — bierze cały plik.
    """
    input_path = os.path.join(DATA_PATH_RAW, "sessions.tsv")
    if not os.path.exists(input_path):
        print(f"Nie znaleziono pliku: {input_path}")
        return

    lengths = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=get_line_count(input_path), desc="Wczytywanie sesji"):
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                try:
                    tracks = json.loads(parts[3])
                    lengths.append(len(tracks))
                except (ValueError, json.JSONDecodeError):
                    continue

    if not lengths:
        print("Brak sesji w pliku.")
        return

    lengths = np.array(lengths)

    print(f"\n{'='*50}")
    print(f" PERCENTYLE DŁUGOŚCI SESJI ({input_path})")
    print(f"{'='*50}")
    print(f"Liczba sesji:  {len(lengths):,}")
    print(f"Min:           {lengths.min()}")
    print(f"Max:           {lengths.max()}")
    print(f"Średnia:       {lengths.mean():.2f}")
    print()
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]:
        print(f"  P{p:5.1f}:  {np.percentile(lengths, p):.0f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # 1. Załaduj raz
    # s_ids, s_pts, s_ts = load_sessions_data()
    
    # # 2. Statystyki w konsoli
    # print_stats(s_pts)
    
    # # 3. Poszczególne zadania
    # plot_seconds_distribution(s_pts, limit=200)
    # plot_hours_distribution(s_pts)
    # analyze_outliers(s_ids, s_pts)
    # plot_sessions_per_day(s_ts)
    
    # 4. Nowa analiza dla sessions.tsv (125-65 dni)
    # analyze_tsv_time_window(125, 65)

    # 5. Percentyle długości sesji
    session_length_percentiles()
    
    print("\nAnaliza zakończona.")



# import json
# import os
# import re
# import shutil
# import argparse
# import subprocess
# from tqdm import tqdm
# from collections import Counter
# from urllib.parse import unquote_plus

# DATA_FILE = "sessions"
# DATA_PATH_RAW = "dataset_raw/"
# DATA_PATH_TEMP = "dataset_temp/"
# DATA_PATH_PROCESSED = "dataset_processed/"

# MIN_TRACK_PLAYCOUNT = 5
# MIN_SESSION_LENGTH = 2
# MAX_SESSION_LENGTH = 90
# MIN_SESSION_PLAYTIME = 30
# MAX_SESSION_PLAYTIME = 1_000_000
# MAX_SESSION_RECENT_TRACKS = MAX_SESSION_LENGTH
# DAYS_FROM_MAX = 365
# DAYS_TO_MAX = 65

# parser = argparse.ArgumentParser()

# parser.add_argument("--min_track_playcount", type=int)
# parser.add_argument("--min_session_length", type=int)
# parser.add_argument("--max_session_length", type=int)
# parser.add_argument("--min_session_playtime", type=int)
# parser.add_argument("--max_session_playtime", type=int)
# parser.add_argument("--max_session_recent_tracks", type=int)
# parser.add_argument("--days_from_max", type=int)
# parser.add_argument("--days_to_max", type=int)

# args = parser.parse_args()

# if args.min_track_playcount is not None:
#     MIN_TRACK_PLAYCOUNT = args.min_track_playcount
# if args.min_session_length is not None:
#     MIN_SESSION_LENGTH = args.min_session_length
# if args.max_session_length is not None:
#     MAX_SESSION_LENGTH = args.max_session_length
# if args.days_from_max is not None:
#     DAYS_FROM_MAX = args.days_from_max
# if args.days_to_max is not None:
#     DAYS_TO_MAX = args.days_to_max
# if args.min_session_playtime is not None:
#     MIN_SESSION_PLAYTIME = args.min_session_playtime
# if args.max_session_playtime is not None:
#     MAX_SESSION_PLAYTIME = args.max_session_playtime
# if args.max_session_recent_tracks is not None:
#     MAX_SESSION_RECENT_TRACKS = args.max_session_recent_tracks

# def copy_processed_to_temp():
#     shutil.copy(
#         get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
#         get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
#     )

# def get_data_file_path(data_path, data_file, file_extension=".tsv"):
#     return os.path.join(data_path, f"{data_file}{file_extension}")

# def get_line_count(file_path):
#     return int(subprocess.check_output(['wc', '-l', file_path]).split()[0])

# def remove_temp_file():
#     temp_file_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
#     if os.path.exists(temp_file_path):
#         os.remove(temp_file_path)

# def initialize():
#     print("\nInitializing data...")
    
#     input_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE, file_extension=".idomaar")
#     output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

#     def get_timestamp(line):
#         try:
#             return int(line[len("event.session\t"):].split("\t")[1])
#         except (IndexError, ValueError):
#             return None

#     raw_lines = []
#     with open(input_path, "r", encoding="utf-8") as fin:
#         for line in tqdm(fin, total=get_line_count(input_path), desc=f"Reading {input_path}"):
#             timestamp = get_timestamp(line)
#             if timestamp is not None:
#                 raw_lines.append((timestamp, line))

#     print(f"Sorting {input_path}")
#     raw_lines.sort(key=lambda x: x[0])

#     with open(output_path, "w", encoding="utf-8") as fout:
#         for _, line in tqdm(raw_lines, desc=f"Writing {output_path}"):
#             line = line[len("event.session\t"):]

#             try:
#                 parts = line.split("\t")
#                 session_id = int(parts[0])
#                 session_timestamp = int(parts[1])
#                 session_stats = json.loads(parts[2][:parts[2].find("} {") + 1])
#                 session_objects = json.loads(line[line.find("} {") + 2:].strip())
#             except (json.JSONDecodeError, ValueError, IndexError):
#                 continue

#             session_playtime = session_stats["playtime"]
#             session_user_id = session_objects["subjects"][0]["id"]
#             session_tracks = [
#                 {
#                     "id": st["id"],
#                     "ps": st["playstart"],
#                     # "pt": st["playtime"],
#                     # "pr": st.get("playratio"),
#                     # "ac": st.get("action")
#                 }
#                 for st in session_objects["objects"]
#             ]

#             if not (MIN_SESSION_PLAYTIME <= session_playtime <= MAX_SESSION_PLAYTIME):
#                 continue

#             fout.write(f"{session_id}\t{session_timestamp}\t{session_user_id}\t{json.dumps(session_tracks, separators=(',', ':'))}\n")

#     shutil.copy(
#         get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
#         get_data_file_path(DATA_PATH_RAW, DATA_FILE)
#     )

# def filter_by_time_window(days_from_max = DAYS_FROM_MAX, days_to_max = DAYS_TO_MAX):
#     if days_from_max == 365 and days_to_max == 0:
#         return

#     print(f"\nFiltering sessions: last {days_from_max} to {days_to_max} days from max timestamp...")

#     input_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
#     output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

#     max_timestamp = 0
#     with open(input_path, "r", encoding="utf-8") as fin:
#         for line in tqdm(fin, total=get_line_count(input_path), desc="Finding max timestamp"):
#             parts = line.strip().split("\t")
#             ts = int(parts[1])
#             if ts > max_timestamp:
#                 max_timestamp = ts

#     lower_bound = max_timestamp - days_from_max * 86400
#     upper_bound = max_timestamp - days_to_max * 86400

#     kept = 0
#     with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
#         for line in tqdm(fin, total=get_line_count(input_path), desc="Filtering sessions"):
#             parts = line.strip().split("\t")
#             ts = int(parts[1])
#             if lower_bound <= ts <= upper_bound:
#                 fout.write(line)
#                 kept += 1

#     print(f"Kept {kept:,} sessions")

# def filter_tracks_by_playcount():
#     print("\nFiltering tracks by allowed tracks...")

#     input_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
#     output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
#     tracks_file = get_data_file_path(DATA_PATH_RAW, "tracks")

#     allowed_tracks = set()
#     with open(tracks_file, "r", encoding="utf-8") as fin:
#         for line in fin:
#             tid = line.strip().split("\t", 1)[0]
#             allowed_tracks.add(tid)

#     print(f"Loaded {len(allowed_tracks):,} allowed tracks from {tracks_file}")

#     with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
#         for line in tqdm(fin, total=get_line_count(input_path), desc=f"Filtering tracks in {input_path}"):
#             parts = line.strip().split("\t")
#             session_tracks = json.loads(parts[3])
#             session_tracks = [t for t in session_tracks if str(t["id"]) in allowed_tracks]

#             if MIN_SESSION_LENGTH <= len(session_tracks) <= MAX_SESSION_LENGTH:
#                 fout.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\t{json.dumps(session_tracks, separators=(',', ':'))}\n")

# # def fill_playratio():
# #     print("\nFilling missing playratios and clipping values...")

# #     input_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
# #     output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

# #     with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
# #         for line in tqdm(fin, total=get_line_count(input_path), desc=f"Processing playratio in {input_path}"):
# #             parts = line.strip().split("\t")
# #             tracks_data = json.loads(parts[3])
            
# #             for t in tracks_data:
# #                 if t["pr"] is None and t["ac"] == "play":
# #                     t["pr"] = 1.0
# #                 elif t["pr"] is None and t["ac"] == "skip":
# #                     t["pr"] = 0.0
# #                 elif t["pr"] is not None and t["pr"] > 2.0 and t["ac"] == "play":
# #                     t["pr"] = min(t["pr"], 2.0)

# #             parts[3] = json.dumps(tracks_data, separators=(',', ':'))
# #             fout.write("\t".join(parts) + "\n")

# def make_inter_file(alias):
#     print("\nCreating .inter file...")
    
#     input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
#     output_path = os.path.join("dataset", alias, f"{alias}.inter")
    
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
#         fout.write("session_id:token\tuser_id:token\titem_id:token\ttimestamp:float\n")
        
#         for line in tqdm(fin, total=get_line_count(input_path), desc=f"Building .inter from {input_path}"):
#             parts = line.strip().split("\t")
#             session_id = parts[0]
#             timestamp = parts[1]
#             user_id = parts[2]
#             tracks = json.loads(parts[3])
            
#             for track in tracks:
#                 fout.write(f"{session_id}\t{user_id}\t{track['id']}\t{int(timestamp) + int(track['ps'])}\n")

# def make_tracks_file(alias):
#     print("\nCreating tracks file...")

#     input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
#     tracks_source = get_data_file_path(DATA_PATH_RAW, "tracks")
#     output_path = os.path.join("dataset", alias, "tracks.tsv")

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     track_ids = set()
#     with open(input_path, "r", encoding="utf-8") as fin:
#         for line in tqdm(fin, total=get_line_count(input_path), desc="Collecting track IDs"):
#             parts = line.strip().split("\t")
#             session_tracks = json.loads(parts[3])
#             for t in session_tracks:
#                 track_ids.add(str(t["id"]))

#     kept = 0
#     seen = set()
#     with open(tracks_source, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
#         for line in tqdm(fin, total=get_line_count(tracks_source), desc="Filtering tracks"):
#             stripped = line.strip()
#             tid = stripped.split("\t", 1)[0]
#             if tid in track_ids and stripped not in seen:
#                 seen.add(stripped)
#                 fout.write(line)
#                 kept += 1

# _UNKNOWN_PATTERNS = re.compile(
#     r'\[unknown\]'
#     r'|<Artista Desconhecido>'
#     r'|<Artista desconocido>'
#     r'|\(artistes? inconnus?\)'
#     r'|<Nieznany wykonawca>'
#     r'|<Bilinmeyen>'
#     r'|<Desconhecido>'
#     r'|<Unbekannter Interpret>'
#     r'|<Okänd artist>'
#     r'|unknown\s*artist'
#     r'|artiste?\s*inconnu'
#     r'|various\s*artists?',
#     re.IGNORECASE
# )

# _URL_PATTERN = re.compile(
#     r'www\.|\.com|\.net|\.org|\.ru|\.info|https?://',
#     re.IGNORECASE
# )

# def _is_noisy(text):
#     if not text or not text.strip():
#         return True
#     if '\ufffd' in text:
#         return True
#     letter_count = sum(1 for c in text if c.isalpha())
#     if letter_count < 2:
#         return True
#     if _UNKNOWN_PATTERNS.search(text):
#         return True
#     if _URL_PATTERN.search(text):
#         return True
#     return False

# def make_track_names_file():
#     print("\nCreating track names file...")

#     tracks_source = os.path.join(DATA_PATH_RAW, "tracks.idomaar")
#     persons_source = os.path.join(DATA_PATH_RAW, "persons.idomaar")
#     output_path = os.path.join(DATA_PATH_RAW, "tracks.tsv")

#     artist_names = {}
#     with open(persons_source, 'r', encoding='utf-8') as f_persons:
#         for line in tqdm(f_persons, total=get_line_count(persons_source), desc="Loading persons"):
#             parts = line.strip().split('\t')
#             artist_id = parts[1]
#             meta = json.loads(parts[3])
#             artist_names[artist_id] = unquote_plus(meta['name'])

#     with open(tracks_source, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
#         for line in tqdm(f_in, total=get_line_count(tracks_source), desc="Processing tracks"):
#             parts = line.strip().split('\t')
#             track_id = parts[1]
#             meta = json.loads(parts[3])
#             name = unquote_plus(meta['name'])

#             relations = json.loads(parts[4])
#             artists = relations.get('artists', [])
#             if artists:
#                 artist_id = str(artists[0]['id'])
#                 artist_name = artist_names.get(artist_id, '')
#             else:
#                 artist_id = ''
#                 artist_name = ''

#             if _is_noisy(name) or _is_noisy(artist_name) or artist_name == '#':
#                 continue

#             if (meta.get('playcount') or 0) >= MIN_TRACK_PLAYCOUNT:  
#                 f_out.write(f"{track_id}\t{name}\t{artist_id}\t{artist_name}\t{meta['playcount']}\n")

# def get_dataset_name():
#     name_parts = [
#         f"days[{DAYS_FROM_MAX}-{DAYS_TO_MAX}]",
#         f"pcount[{MIN_TRACK_PLAYCOUNT}]",
#         f"ptime[{MIN_SESSION_PLAYTIME}-{MAX_SESSION_PLAYTIME}]",
#         f"length[{MIN_SESSION_LENGTH}-{MAX_SESSION_LENGTH}]",
#         f"recent[{MAX_SESSION_RECENT_TRACKS}]",
#     ]

#     return "30music__" + "_".join(name_parts)

# if __name__ == "__main__":
#     os.makedirs(DATA_PATH_TEMP, exist_ok=True)
#     os.makedirs(DATA_PATH_PROCESSED, exist_ok=True)

#     if not os.path.exists(get_data_file_path(DATA_PATH_RAW, DATA_FILE)):
#         initialize()

#     if not os.path.exists(get_data_file_path(DATA_PATH_RAW, "tracks")):
#         make_track_names_file()

#     filter_by_time_window()
#     copy_processed_to_temp()
#     filter_tracks_by_playcount()
#     copy_processed_to_temp()
#     # fill_playratio()
#     make_inter_file(get_dataset_name())
#     make_tracks_file(get_dataset_name())
#     remove_temp_file()