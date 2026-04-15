import json
import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

DATA_PATH_RAW = "dataset_raw/"
DATA_PATH_TEMP = "dataset_temp/"
DATA_PATH_PROCESSED = "dataset_processed/"
DATA_FILE = "sessions"

MIN_TRACK_PLAYCOUNT = 10
MIN_SESSION_LENGTH = 2
DAYS_FROM_MAX = 180
DAYS_TO_MAX = 0

parser = argparse.ArgumentParser()

parser.add_argument("--min_track_playcount", type=int)
parser.add_argument("--min_session_length", type=int)
parser.add_argument("--days_from_max", type=int)
parser.add_argument("--days_to_max", type=int)

args = parser.parse_args()

if args.min_track_playcount is not None:
    MIN_TRACK_PLAYCOUNT = args.min_track_playcount

if args.min_session_length is not None:
    MIN_SESSION_LENGTH = args.min_session_length

if args.days_from_max is not None:
    DAYS_FROM_MAX = args.days_from_max

if args.days_to_max is not None:
    DAYS_TO_MAX = args.days_to_max


def get_data_file_path(data_path, data_file, file_extension=".tsv"):
    return os.path.join(data_path, f"{data_file}{file_extension}")

def get_line_count(file_path): 
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def remove_temp_file():
    temp_file_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

def initialize():
    print("\nInitializing data...")
    
    input_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE, file_extension=".idomaar")
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

    def get_timestamp(line):
        try:
            return int(line[len("event.session\t"):].split("\t")[1])
        except (IndexError, ValueError):
            return None

    raw_lines = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=get_line_count(input_path), desc=f"Reading {input_path}"):
            timestamp = get_timestamp(line)
            if timestamp is not None:
                raw_lines.append((timestamp, line))

    print(f"Sorting {input_path}")
    raw_lines.sort(key=lambda x: x[0])

    with open(output_path, "w", encoding="utf-8") as fout:
        for _, line in tqdm(raw_lines, desc=f"Writing {output_path}"):
            line = line[len("event.session\t"):]

            try:
                parts = line.split("\t")
                session_id = int(parts[0])
                session_timestamp = int(parts[1])
                session_objects = json.loads(line[line.find("} {") + 2:].strip())
            except (json.JSONDecodeError, ValueError, IndexError):
                continue

            session_user_id = session_objects["subjects"][0]["id"]
            session_tracks = [
                {
                    "id": st["id"],
                    "ps": st["playstart"],
                    "pt": st["playtime"],
                    "pr": st.get("playratio"),
                    # "pr": 0 if st.get("action") == "skip" and st.get("playratio") is None else st.get("playratio")
                }
                for st in session_objects["objects"]
                # if st.get("action") == "play"
            ]

            # if len(session_tracks) < MIN_SESSION_LENGTH:
            #     continue

            fout.write(f"{session_id}\t{session_timestamp}\t{session_user_id}\t{json.dumps(session_tracks, separators=(',', ':'))}\n")

def filter_by_time_window(days_from_max: int, days_to_max: int = 0):
    if days_from_max == 365:
        return

    print(f"\nFiltering sessions: last {days_from_max} to {days_to_max} days from max timestamp...")

    input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)

    max_timestamp = 0
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=get_line_count(input_path), desc="Finding max timestamp"):
            parts = line.strip().split("\t")
            ts = int(parts[1])
            if ts > max_timestamp:
                max_timestamp = ts

    lower_bound = max_timestamp - days_from_max * 86400
    upper_bound = max_timestamp - days_to_max * 86400

    kept = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=get_line_count(input_path), desc="Filtering sessions"):
            parts = line.strip().split("\t")
            ts = int(parts[1])
            if lower_bound <= ts <= upper_bound:
                fout.write(line)
                kept += 1

    print(f"Kept {kept:,} sessions")

def filter_tracks_by_playcount():
    print("\nFiltering tracks by playcount...")

    input_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    track_counts = Counter()

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=get_line_count(input_path), desc=f"Counting tracks in {input_path}"):
            parts = line.strip().split("\t")
            session_tracks = json.loads(parts[3])
            for t in session_tracks:
                track_counts[t["id"]] += 1

    allowed_tracks = {tid for tid, count in track_counts.items() if count >= MIN_TRACK_PLAYCOUNT}

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=get_line_count(input_path), desc=f"Filtering tracks in {input_path}"):
            parts = line.strip().split("\t")
            session_tracks = json.loads(parts[3])
            session_tracks = [t for t in session_tracks if t["id"] in allowed_tracks]

            if len(session_tracks) >= MIN_SESSION_LENGTH:
                fout.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\t{json.dumps(session_tracks, separators=(',', ':'))}\n")

def fill_playratio():
    print("\nFilling missing playratios and clipping values...")

    input_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

    null_count = 0
    clipped_count = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=get_line_count(input_path), desc=f"Processing playratio in {input_path}"):
            parts = line.strip().split("\t")
            tracks_data = json.loads(parts[3])
            
            for t in tracks_data:
                if t["pr"] is None:
                    t["pr"] = 1.0
                    null_count += 1
                if t["pr"] > 2.0:
                    clipped_count += 1
                t["pr"] = min(t["pr"], 2.0)
            
            parts[3] = json.dumps(tracks_data, separators=(',', ':'))
            fout.write("\t".join(parts) + "\n")

    # print(f"Replaced nulls with 1.0: {null_count:,}")
    # print(f"Clipped values to 2.0: {clipped_count:,}")

def analyze_dataset():
    import numpy as np
    
    print("\nAnalyzing dataset statistics...")
    
    input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    
    if not os.path.exists(input_path):
        print(f"Processed file {input_path} not found. Skipping analysis.")
        return

    track_counts = Counter()
    session_lengths = []
    user_session_counts = Counter()
    user_interaction_counts = Counter()
    playratios = []
    total_sessions = 0
    total_interactions = 0
    
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=get_line_count(input_path), desc="Analyzing dataset"):
            parts = line.strip().split("\t")
            user_id = parts[2]
            tracks = json.loads(parts[3])
            
            user_session_counts[user_id] += 1
            user_interaction_counts[user_id] += len(tracks)
            session_lengths.append(len(tracks))
            total_sessions += 1
            total_interactions += len(tracks)
            
            for t in tracks:
                track_counts[t["id"]] += 1
                pr = t.get("pr")
                if pr is not None:
                    playratios.append(float(pr))

    unique_tracks = len(track_counts)
    unique_users = len(user_session_counts)
    
    session_lengths_arr = np.array(session_lengths)
    track_counts_arr = np.array(sorted(track_counts.values(), reverse=True))
    user_sessions_arr = np.array(sorted(user_session_counts.values(), reverse=True))
    user_inters_arr = np.array(sorted(user_interaction_counts.values(), reverse=True))
    playratios_arr = np.array(playratios) if playratios else np.array([])
    
    sparsity = 1.0 - (total_interactions / (unique_users * unique_tracks)) if unique_users * unique_tracks > 0 else 0
    
    os.makedirs("analysis", exist_ok=True)
    report_file = f"analysis/stats_t{MIN_TRACK_PLAYCOUNT}_sl{MIN_SESSION_LENGTH}_d{DAYS_FROM_MAX}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"=== Dataset Statistics ===\n")
        f.write(f"Parameters: min_track_playcount={MIN_TRACK_PLAYCOUNT}, min_session_length={MIN_SESSION_LENGTH}, days_from_max={DAYS_FROM_MAX}\n\n")
        
        f.write(f"Total sessions: {total_sessions}\n")
        f.write(f"Total unique users: {unique_users}\n")
        f.write(f"Total unique tracks: {unique_tracks}\n")
        f.write(f"Total interactions: {total_interactions}\n")
        f.write(f"Sparsity: {sparsity*100:.4f}%\n\n")
        
        if total_sessions > 0:
            f.write(f"--- Session lengths ---\n")
            f.write(f"  Mean: {session_lengths_arr.mean():.2f}\n")
            f.write(f"  Median: {np.median(session_lengths_arr):.1f}\n")
            f.write(f"  Std: {session_lengths_arr.std():.2f}\n")
            f.write(f"  Min: {session_lengths_arr.min()}\n")
            f.write(f"  Max: {session_lengths_arr.max()}\n")
            f.write(f"  P25: {np.percentile(session_lengths_arr, 25):.1f}\n")
            f.write(f"  P75: {np.percentile(session_lengths_arr, 75):.1f}\n")
            f.write(f"  P95: {np.percentile(session_lengths_arr, 95):.1f}\n\n")
        
        if unique_tracks > 0:
            f.write(f"--- Track playcounts ---\n")
            f.write(f"  Mean: {track_counts_arr.mean():.2f}\n")
            f.write(f"  Median: {np.median(track_counts_arr):.1f}\n")
            f.write(f"  Std: {track_counts_arr.std():.2f}\n")
            f.write(f"  Min: {track_counts_arr.min()}\n")
            f.write(f"  Max: {track_counts_arr.max()}\n")
            f.write(f"  P25: {np.percentile(track_counts_arr, 25):.1f}\n")
            f.write(f"  P75: {np.percentile(track_counts_arr, 75):.1f}\n")
            f.write(f"  P95: {np.percentile(track_counts_arr, 95):.1f}\n\n")
        
        if unique_users > 0:
            f.write(f"--- User activity (sessions per user) ---\n")
            f.write(f"  Mean: {user_sessions_arr.mean():.2f}\n")
            f.write(f"  Median: {np.median(user_sessions_arr):.1f}\n")
            f.write(f"  Min: {user_sessions_arr.min()}\n")
            f.write(f"  Max: {user_sessions_arr.max()}\n")
            f.write(f"  P25: {np.percentile(user_sessions_arr, 25):.1f}\n")
            f.write(f"  P75: {np.percentile(user_sessions_arr, 75):.1f}\n")
            f.write(f"  P95: {np.percentile(user_sessions_arr, 95):.1f}\n\n")

            f.write(f"--- User activity (interactions per user) ---\n")
            f.write(f"  Mean: {user_inters_arr.mean():.2f}\n")
            f.write(f"  Median: {np.median(user_inters_arr):.1f}\n")
            f.write(f"  Min: {user_inters_arr.min()}\n")
            f.write(f"  Max: {user_inters_arr.max()}\n")
            f.write(f"  P25: {np.percentile(user_inters_arr, 25):.1f}\n")
            f.write(f"  P75: {np.percentile(user_inters_arr, 75):.1f}\n")
            f.write(f"  P95: {np.percentile(user_inters_arr, 95):.1f}\n\n")
        
        # if len(playratios_arr) > 0:
        #     f.write(f"--- Playratio ---\n")
        #     f.write(f"  Count (non-null): {len(playratios_arr)}\n")
        #     f.write(f"  Mean: {playratios_arr.mean():.4f}\n")
        #     f.write(f"  Median: {np.median(playratios_arr):.4f}\n")
        #     f.write(f"  Std: {playratios_arr.std():.4f}\n")
        #     f.write(f"  Ratio with pr=0 (skips): {(playratios_arr == 0).sum()} ({(playratios_arr == 0).sum()/len(playratios_arr)*100:.2f}%)\n")
        #     f.write(f"  Ratio with pr>=1 (full listen): {(playratios_arr >= 1.0).sum()} ({(playratios_arr >= 1.0).sum()/len(playratios_arr)*100:.2f}%)\n\n")

    print(f"Analysis stats saved to {report_file}")
            
    if unique_tracks == 0 or total_sessions == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(track_counts_arr)
    ax.set_yscale('log')
    ax.set_xlabel('Track Rank')
    ax.set_ylabel('Playcount (log scale)')
    ax.set_title('Track Popularity Distribution')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    sl_counts = Counter(session_lengths)
    lengths, freqs = zip(*sorted(sl_counts.items()))
    ax.bar(lengths, freqs, color='steelblue', edgecolor='none')
    ax.set_xlim(left=0, right=min(50, max(lengths) + 1))
    ax.set_xlabel('Session Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Session Length Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 0]
    ax.plot(user_sessions_arr, color='darkorange')
    ax.set_yscale('log')
    ax.set_xlabel('User Rank')
    ax.set_ylabel('Sessions per User (log scale)')
    ax.set_title('User Activity Distribution')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(playratios_arr) > 0:
        ax.hist(playratios_arr[playratios_arr <= 2.0], bins=50, color='seagreen', edgecolor='none', alpha=0.8)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, label='Full listen (pr=1.0)')
        ax.set_xlabel('Playratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Playratio Distribution (clipped to 2.0)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No playratio data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Playratio Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(
        f"Dataset Stats (MinTrack:{MIN_TRACK_PLAYCOUNT}, MinSess:{MIN_SESSION_LENGTH}, Days:{DAYS_FROM_MAX})\n"
        f"Sessions: {total_sessions:,} | Users: {unique_users:,} | Tracks: {unique_tracks:,} | "
        f"Interactions: {total_interactions:,} | Sparsity: {sparsity*100:.2f}%",
        fontsize=11
    )
    
    plt.tight_layout()
    plot_file = f"analysis/stats_t{MIN_TRACK_PLAYCOUNT}_sl{MIN_SESSION_LENGTH}_d{DAYS_FROM_MAX}.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
   
    print(f"Analysis plot saved to {plot_file}")

def make_inter_file():
    print("\nCreating .inter file...")
    
    input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    output_path = os.path.join("dataset", "30music", "30music.inter")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        fout.write("session_id:token\tuser_id:token\titem_id:token\ttimestamp:float\trating:float\n")
        
        for line in tqdm(fin, total=get_line_count(input_path), desc=f"Building .inter from {input_path}"):
            parts = line.strip().split("\t")
            session_id = parts[0]
            timestamp = parts[1]
            user_id = parts[2]
            tracks = json.loads(parts[3])
            
            for track in tracks:
                fout.write(f"{session_id}\t{user_id}\t{track['id']}\t{int(timestamp) + int(track['ps'])}\t{track['pr']}\n")

if __name__ == "__main__":
    os.makedirs(DATA_PATH_TEMP, exist_ok=True)
    os.makedirs(DATA_PATH_PROCESSED, exist_ok=True)
    
    # wstępne czyszczenie
    initialize()
    
    shutil.copy(
        get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
        get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    )

    # filtrowanie wg okna czasowego
    filter_by_time_window(days_from_max=DAYS_FROM_MAX, days_to_max=DAYS_TO_MAX)
    
    # filtrowanie wg liczby odsłuchań
    filter_tracks_by_playcount()
    
    # shutil.copy(
    #     get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
    #     get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    # )
    
    # # uzupełnianie playratio
    # fill_playratio()
    
    # # tworzenie pliku .inter
    # make_inter_file()
    
    # # usuwanie plików tymczasowych
    # remove_temp_file()
    
    # analiza zbioru danych po przetworzeniu
    analyze_dataset()
