import json
import os
import shutil
import argparse
import subprocess
from tqdm import tqdm
from collections import Counter
from urllib.parse import unquote_plus

DATA_FILE = "sessions"
DATA_PATH_RAW = "dataset_raw/"
DATA_PATH_TEMP = "dataset_temp/"
DATA_PATH_PROCESSED = "dataset_processed/"

MIN_TRACK_PLAYCOUNT = 5
MIN_SESSION_LENGTH = 2
MAX_SESSION_LENGTH = 90
MIN_SESSION_PLAYTIME = 30
MAX_SESSION_PLAYTIME = 1_000_000
MAX_SESSION_RECENT_TRACKS = MAX_SESSION_LENGTH
DAYS_FROM_MAX = 365
DAYS_TO_MAX = 65

parser = argparse.ArgumentParser()

parser.add_argument("--min_track_playcount", type=int)
parser.add_argument("--min_session_length", type=int)
parser.add_argument("--max_session_length", type=int)
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
if args.max_session_length is not None:
    MAX_SESSION_LENGTH = args.max_session_length
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
                    # "pt": st["playtime"],
                    # "pr": st.get("playratio"),
                    # "ac": st.get("action")
                }
                for st in session_objects["objects"][-(MAX_SESSION_RECENT_TRACKS + 1):-1]
            ]

            if not (MIN_SESSION_PLAYTIME <= session_playtime <= MAX_SESSION_PLAYTIME):
                continue

            fout.write(f"{session_id}\t{session_timestamp}\t{session_user_id}\t{json.dumps(session_tracks, separators=(',', ':'))}\n")

    shutil.copy(
        get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
        get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    )

def filter_by_time_window(days_from_max = DAYS_FROM_MAX, days_to_max = DAYS_TO_MAX):
    if days_from_max == 365 and days_to_max == 0:
        return

    print(f"\nFiltering sessions: last {days_from_max} to {days_to_max} days from max timestamp...")

    input_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
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

            if MIN_SESSION_LENGTH <= len(session_tracks) <= MAX_SESSION_LENGTH:
                fout.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\t{json.dumps(session_tracks, separators=(',', ':'))}\n")

def fill_playratio():
    print("\nFilling missing playratios and clipping values...")

    input_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=get_line_count(input_path), desc=f"Processing playratio in {input_path}"):
            parts = line.strip().split("\t")
            tracks_data = json.loads(parts[3])
            
            for t in tracks_data:
                if t["pr"] is None and t["ac"] == "play":
                    t["pr"] = 1.0
                elif t["pr"] is None and t["ac"] == "skip":
                    t["pr"] = 0.0
                elif t["pr"] is not None and t["pr"] > 2.0 and t["ac"] == "play":
                    t["pr"] = min(t["pr"], 2.0)

            parts[3] = json.dumps(tracks_data, separators=(',', ':'))
            fout.write("\t".join(parts) + "\n")

def make_inter_file(alias):
    print("\nCreating .inter file...")
    
    input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    output_path = os.path.join("dataset", alias, f"{alias}.inter")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        fout.write("session_id:token\tuser_id:token\titem_id:token\ttimestamp:float\n")
        
        for line in tqdm(fin, total=get_line_count(input_path), desc=f"Building .inter from {input_path}"):
            parts = line.strip().split("\t")
            session_id = parts[0]
            timestamp = parts[1]
            user_id = parts[2]
            tracks = json.loads(parts[3])
            
            for track in tracks:
                fout.write(f"{session_id}\t{user_id}\t{track['id']}\t{int(timestamp) + int(track['ps'])}\n")

def make_tracks_file(alias):
    print("\nCreating tracks file...")

    input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    tracks_source = get_data_file_path(DATA_PATH_RAW, "tracks")
    output_path = os.path.join("dataset", alias, "tracks.tsv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    track_ids = set()
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=get_line_count(input_path), desc="Collecting track IDs"):
            parts = line.strip().split("\t")
            session_tracks = json.loads(parts[3])
            for t in session_tracks:
                track_ids.add(str(t["id"]))

    kept = 0
    seen = set()
    with open(tracks_source, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=get_line_count(tracks_source), desc="Filtering tracks"):
            stripped = line.strip()
            tid = stripped.split("\t", 1)[0]
            if tid in track_ids and stripped not in seen:
                seen.add(stripped)
                fout.write(line)
                kept += 1

def make_track_names_file():
    print("\nCreating track names file...")
    
    tracks_source = os.path.join(DATA_PATH_RAW, "tracks.idomaar")
    output_path = os.path.join(DATA_PATH_RAW, "tracks.tsv")
    
    with open(tracks_source, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, total=get_line_count(tracks_source), desc="Filtering tracks"):
            parts = line.strip().split('\t')
            track_id = parts[1]
            meta = json.loads(parts[3])
            name = unquote_plus(meta['name'])
            f_out.write(f"{track_id}\t{name}\n")

def get_dataset_name():
    name_parts = [
        f"days[{DAYS_FROM_MAX}-{DAYS_TO_MAX}]",
        f"pcount[{MIN_TRACK_PLAYCOUNT}]",
        f"ptime[{MIN_SESSION_PLAYTIME}-{MAX_SESSION_PLAYTIME}]",
        f"length[{MIN_SESSION_LENGTH}-{MAX_SESSION_LENGTH}]",
        f"recent[{MAX_SESSION_RECENT_TRACKS}]",
    ]

    return "30music__" + "_".join(name_parts)

if __name__ == "__main__":
    os.makedirs(DATA_PATH_TEMP, exist_ok=True)
    os.makedirs(DATA_PATH_PROCESSED, exist_ok=True)

    if not os.path.exists(get_data_file_path(DATA_PATH_RAW, DATA_FILE)):
        initialize()

    if not os.path.exists(get_data_file_path(DATA_PATH_RAW, "tracks")):
        make_track_names_file()

    filter_by_time_window()
    copy_processed_to_temp()
    filter_tracks_by_playcount()
    copy_processed_to_temp()
    # fill_playratio()
    make_inter_file(get_dataset_name())
    make_tracks_file(get_dataset_name())
    remove_temp_file()