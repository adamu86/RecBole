import json
import os
import shutil
import argparse
import subprocess
from tqdm import tqdm
from collections import Counter

DATA_FILE = "sessions"
DATA_PATH_RAW = "dataset_raw/"
DATA_PATH_TEMP = "dataset_temp/"
DATA_PATH_PROCESSED = "dataset_processed/"

MIN_TRACK_PLAYCOUNT = 5
MIN_SESSION_LENGTH = 2
MIN_SESSION_PLAYTIME = 10
MAX_SESSION_PLAYTIME = 1_000_000
MAX_SESSION_RECENT_TRACKS = 50
DAYS_FROM_MAX = 180
DAYS_TO_MAX = 90

parser = argparse.ArgumentParser()

parser.add_argument("--min_track_playcount", type=int)
parser.add_argument("--min_session_length", type=int)
parser.add_argument("--min_session_playtime", type=int)
parser.add_argument("--max_session_playtime", type=int)
parser.add_argument("--max_session_recent_tracks", type=int)
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

if args.min_session_playtime is not None:
    MIN_SESSION_PLAYTIME = args.min_session_playtime

if args.max_session_playtime is not None:
    MAX_SESSION_PLAYTIME = args.max_session_playtime

if args.max_session_recent_tracks is not None:
    MAX_SESSION_RECENT_TRACKS = args.max_session_recent_tracks

def copy_processed_to_temp():
    shutil.copy(
        get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
        get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    )

def get_data_file_path(data_path, data_file, file_extension=".tsv"):
    return os.path.join(data_path, f"{data_file}{file_extension}")

def get_line_count(file_path):
    return int(subprocess.check_output(['wc', '-l', file_path]).split()[0])

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
                session_stats = json.loads(parts[2][:parts[2].find("} {") + 1])
                session_objects = json.loads(line[line.find("} {") + 2:].strip())
            except (json.JSONDecodeError, ValueError, IndexError):
                continue

            session_playtime = session_stats["playtime"]
            session_user_id = session_objects["subjects"][0]["id"]
            session_tracks = [
                {
                    "id": st["id"],
                    "ps": st["playstart"],
                    "pt": st["playtime"],
                    "pr": st.get("playratio"),
                    "ac": st.get("action")
                }
                for st in session_objects["objects"][-(MAX_SESSION_RECENT_TRACKS + 1):-1]
            ]

            if not (MIN_SESSION_PLAYTIME <= session_playtime <= MAX_SESSION_PLAYTIME):
                continue

            fout.write(f"{session_id}\t{session_timestamp}\t{session_user_id}\t{json.dumps(session_tracks, separators=(',', ':'))}\n")

def filter_by_time_window(days_from_max: int, days_to_max: int = 0):
    if days_from_max == 365:
        return

    print(f"\nFiltering sessions: last {days_from_max} to {days_to_max} days from max timestamp...")

    input_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

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

def make_inter_file(alias):
    print("\nCreating .inter file...")
    
    input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    output_path = os.path.join("dataset", alias, f"{alias}.inter")
    
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
    copy_processed_to_temp()

    # filtrowanie wg okna czasowego
    filter_by_time_window(days_from_max=DAYS_FROM_MAX, days_to_max=DAYS_TO_MAX)
    copy_processed_to_temp()

    # filtrowanie wg liczby odsłuchań
    filter_tracks_by_playcount()
    copy_processed_to_temp()
    
    # uzupełnianie playratio
    fill_playratio()
    
    # tworzenie pliku .inter
    make_inter_file("30music")
    
    # usuwanie plików tymczasowych
    remove_temp_file()