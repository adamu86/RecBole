import json
import os
import shutil
from collections import Counter
from tqdm import tqdm

DATA_PATH_RAW = "dataset_raw/"
DATA_PATH_TEMP = "dataset_temp/"
DATA_PATH_PROCESSED = "dataset_processed/"
DATA_FILE = "sessions"

MIN_TRACK_PLAYCOUNT = 2
MIN_SESSION_LENGTH = 2

DAYS_FROM_MAX = 5
DAYS_TO_MAX = 0

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
                    "pr": st.get("playratio")
                }
                for st in session_objects["objects"]
                if st.get("action") == "play"
            ]

            fout.write(f"{session_id}\t{session_timestamp}\t{session_user_id}\t{json.dumps(session_tracks, separators=(',', ':'))}\n")

def filter_by_time_window(days_from_max: int, days_to_max: int = 0):
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

    print(f"Max timestamp: {max_timestamp}")
    print(f"Window: {lower_bound} to {upper_bound}")

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

    print(f"Replaced nulls with 1.0: {null_count:,}")
    print(f"Clipped values to 2.0: {clipped_count:,}")

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
    
    initialize()
    
    shutil.copy(
        get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
        get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    )

    filter_by_time_window(days_from_max=DAYS_FROM_MAX, days_to_max=DAYS_TO_MAX)
    
    filter_tracks_by_playcount()
    
    shutil.copy(
        get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
        get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    )
    
    fill_playratio()
    
    make_inter_file()
    
    remove_temp_file()
