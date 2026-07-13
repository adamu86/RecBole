import json
import os
import re
import shutil
import argparse
import subprocess
from tqdm import tqdm
from collections import Counter
from urllib.parse import unquote_plus
import pandas as pd
import numpy as np

DATA_FILE = "sessions"
DATA_PATH_RAW = "dataset_raw/"
DATA_PATH_TEMP = "dataset_temp/"
DATA_PATH_PROCESSED = "dataset_processed/"

MIN_TRACK_PLAYCOUNT = 25
MIN_SESSION_LENGTH = 2
MAX_SESSION_LENGTH = 100
MIN_SESSION_PLAYTIME = 60
MAX_SESSION_PLAYTIME = 1_000_000
MAX_SESSION_RECENT_TRACKS = MAX_SESSION_LENGTH
DAYS_FROM_MAX = 365
DAYS_TO_MAX = 65

MAX_VALID_TRACK_ID = 3893303
MIN_TIMESTAMP = 1390209860
MAX_TIMESTAMP = 1421745720


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

def safe_copy(src, dst):
    if os.path.exists(dst):
        try:
            import stat
            os.chmod(dst, stat.S_IWRITE)
            os.remove(dst)
        except Exception:
            pass
    shutil.copyfile(src, dst)

def copy_processed_to_temp():
    safe_copy(
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

            session_user_id = session_objects["subjects"][0]["id"]
            raw_tracks = []
            last_seen_ps = {}
            for st in session_objects["objects"]:
                track_id = st["id"]
                ps = st["playstart"]
                if track_id in last_seen_ps and abs(ps - last_seen_ps[track_id]) < 10:
                    continue
                last_seen_ps[track_id] = ps
                raw_tracks.append({"id": track_id, "ps": ps})

            if not raw_tracks:
                continue

            # Sort tracks chronologically
            raw_tracks.sort(key=lambda x: x["ps"])
            
            sub_sessions = []
            current_sub = [raw_tracks[0]]
            
            # Split if inactivity > 30 minutes (1800 seconds)
            for i in range(1, len(raw_tracks)):
                curr_track = raw_tracks[i]
                prev_track = current_sub[-1]
                
                if curr_track["ps"] - prev_track["ps"] > 1800:
                    sub_sessions.append(current_sub)
                    current_sub = [curr_track]
                else:
                    current_sub.append(curr_track)
            sub_sessions.append(current_sub)

            # Write out valid sub-sessions
            for sub_idx, sub_session in enumerate(sub_sessions):
                if not (MIN_SESSION_LENGTH <= len(sub_session) <= MAX_SESSION_LENGTH):
                    continue
                
                # Re-calculate playtime for the sub-session
                playtime = sub_session[-1]["ps"] - sub_session[0]["ps"]
                if not (MIN_SESSION_PLAYTIME <= playtime <= MAX_SESSION_PLAYTIME):
                    continue

                new_session_id = f"{session_id}_{sub_idx}" if len(sub_sessions) > 1 else str(session_id)
                fout.write(f"{new_session_id}\t{session_timestamp}\t{session_user_id}\t{json.dumps(sub_session, separators=(',', ':'))}\n")

    safe_copy(
        get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
        get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    )

def filter_by_time_window(days_from_max = DAYS_FROM_MAX, days_to_max = DAYS_TO_MAX):
    print(f"\nFiltering sessions: last {days_from_max} to {days_to_max} days from max timestamp...")

    input_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

    lower_bound = MAX_TIMESTAMP - days_from_max * 86400
    upper_bound = MAX_TIMESTAMP - days_to_max * 86400

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
    print("\nFiltering tracks by tracks.tsv whitelist...")

    input_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    tracks_file = get_data_file_path(DATA_PATH_RAW, "tracks")

    whitelisted = set()
    with open(tracks_file, "r", encoding="utf-8") as fin:
        for line in fin:
            tid = line.strip().split("\t", 1)[0]
            whitelisted.add(int(tid))

    print(f"Loaded {len(whitelisted):,} whitelisted tracks from {tracks_file}")

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=get_line_count(input_path), desc=f"Filtering tracks in {input_path}"):
            parts = line.strip().split("\t")
            session_tracks = json.loads(parts[3])
            session_tracks = [t for t in session_tracks if t["id"] in whitelisted]

            if MIN_SESSION_LENGTH <= len(session_tracks) <= MAX_SESSION_LENGTH:
                fout.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\t{json.dumps(session_tracks, separators=(',', ':'))}\n")
                
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

_UNKNOWN_PATTERNS = re.compile(
    r'\[unknown\]'
    r'|<Artista Desconhecido>'
    r'|<Artista desconocido>'
    r'|\(artistes? inconnus?\)'
    r'|<Nieznany wykonawca>'
    r'|<Bilinmeyen>'
    r'|<Desconhecido>'
    r'|<Unbekannter Interpret>'
    r'|<Okänd artist>'
    r'|unknown\s*artist'
    r'|artiste?\s*inconnu'
    r'|various\s*artists?',
    re.IGNORECASE
)

_URL_PATTERN = re.compile(
    r'www\.|\.com|\.net|\.org|\.ru|\.info|https?://',
    re.IGNORECASE
)

def _is_noisy(text):
    if not text or not text.strip():
        return True
    if '\ufffd' in text:
        return True
    letter_count = sum(1 for c in text if c.isalpha())
    if letter_count < 2:
        return True
    if _UNKNOWN_PATTERNS.search(text):
        return True
    if _URL_PATTERN.search(text):
        return True
    return False

def make_track_names_file():
    print("\nCreating track names file...")

    sessions_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    tracks_source = os.path.join(DATA_PATH_RAW, "tracks.idomaar")
    output_path = os.path.join(DATA_PATH_RAW, "tracks.tsv")

    track_counts = Counter()
    with open(sessions_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=get_line_count(sessions_path), desc="Counting tracks in sessions"):
            parts = line.strip().split("\t")
            session_tracks = json.loads(parts[3])
            for t in session_tracks:
                if int(t["id"]) <= MAX_VALID_TRACK_ID:
                    track_counts[int(t["id"])] += 1

    track_ids = {tid for tid, count in track_counts.items() if count >= MIN_TRACK_PLAYCOUNT}
    print(f"Found {len(track_counts):,} unique tracks, {len(track_ids):,} with >= {MIN_TRACK_PLAYCOUNT} plays")

    tracks = {}
    with open(tracks_source, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=get_line_count(tracks_source), desc="Loading track names"):
            parts = line.strip().split("\t")
            track_id = int(parts[1])

            if track_id not in track_ids:
                continue

            meta = json.loads(parts[3])
            name = unquote_plus(meta["name"])

            if _is_noisy(name):
                continue

            tracks[track_id] = name

    with open(output_path, "w", encoding="utf-8") as fout:
        for track_id, name in sorted(tracks.items()):
            fout.write(f"{track_id}\t{name}\n")

    print(f"Saved {len(tracks):,} clean tracks to {output_path}")

def make_item_file(alias):
    print("\nCreating .item file...")
    
    artist_tags_path = os.path.join("dataset", "artists_tags.tsv")
    tracks_path = os.path.join("dataset", alias, "tracks.tsv")
    output_path = os.path.join("dataset", alias, f"{alias}.item")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    artist_tags = {}
    if os.path.exists(artist_tags_path):
        with open(artist_tags_path, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip('\n').split("\t")
                if len(parts) >= 4:
                    track_id = parts[0]
                    try:
                        lastfm_json = json.loads(parts[2])
                        mb_json = json.loads(parts[3])
                        
                        tags = []
                        if lastfm_json and isinstance(lastfm_json[0], dict):
                            tags.extend([str(t.get("tag", "")).replace(" ", "-") for t in lastfm_json if "tag" in t])
                        
                        if mb_json:
                            if isinstance(mb_json[0], dict):
                                tags.extend([str(t.get("tag", "")).replace(" ", "-") for t in mb_json if "tag" in t])
                            else:
                                tags.extend([str(t).replace(" ", "-") for t in mb_json])
                                
                        seen = set()
                        unique_tags = []
                        for tag in tags:
                            if tag not in seen and tag:
                                seen.add(tag)
                                unique_tags.append(tag)
                                
                        artist_tags[track_id] = " ".join(unique_tags)
                    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                        pass
                elif len(parts) == 3:
                    track_id = parts[0]
                    try:
                        tags_json = json.loads(parts[2])
                        if tags_json and isinstance(tags_json[0], dict):
                            tags = [str(t.get("tag", "")).replace(" ", "-") for t in tags_json if "tag" in t]
                        else:
                            tags = [str(t).replace(" ", "-") for t in tags_json]
                        artist_tags[track_id] = " ".join(tags)
                    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                        pass
                elif len(parts) == 2:
                    artist = parts[0]
                    try:
                        tags_json = json.loads(parts[1])
                        if tags_json and isinstance(tags_json[0], dict):
                            tags = [str(t.get("tag", "")).replace(" ", "-") for t in tags_json if "tag" in t]
                        else:
                            tags = [str(t).replace(" ", "-") for t in tags_json]
                        artist_tags[artist] = " ".join(tags)
                    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                        pass
    else:
        print(f"Warning: {artist_tags_path} not found. All items will have 'unknown' tags.")

    try:
        total_tracks = get_line_count(tracks_path)
    except Exception:
        total_tracks = None

    with open(tracks_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        fout.write("item_id:token\titem_tags:token_seq\n")
        
        for line in tqdm(fin, total=total_tracks, desc=f"Building .item from {tracks_path}"):
            parts = line.strip('\n').split("\t")
            if len(parts) >= 2:
                track_id = parts[0]
                track_info = parts[1]
                
                info_parts = track_info.split("/_/")
                artist_name = info_parts[0]
                
                tags = artist_tags.get(track_id)
                if tags is None:
                    tags = artist_tags.get(artist_name, "")
                    
                if not tags:
                    tags = "unknown"
                
                fout.write(f"{track_id}\t{tags}\n")

def get_dataset_name(prefix="30music__"):
    name_parts = [
        f"days[{DAYS_FROM_MAX}-{DAYS_TO_MAX}]",
        f"pcount[{MIN_TRACK_PLAYCOUNT}]",
        f"ptime[{MIN_SESSION_PLAYTIME}-{MAX_SESSION_PLAYTIME}]",
        f"length[{MIN_SESSION_LENGTH}-{MAX_SESSION_LENGTH}]",
        f"recent[{MAX_SESSION_RECENT_TRACKS}]",
    ]

    return prefix + "_".join(name_parts)

def split_sessions_temporal(df, session_field, time_field, ratios):
    """Split entire sessions by temporal order."""
    session_start_times = df.groupby(session_field)[time_field].min()
    session_start_times = session_start_times.sort_values()
    session_ids_sorted = session_start_times.index.values

    n_sessions = len(session_ids_sorted)
    n_train = int(n_sessions * ratios[0])
    n_valid = int(n_sessions * ratios[1])

    train_sessions = set(session_ids_sorted[:n_train])
    valid_sessions = set(session_ids_sorted[n_train : n_train + n_valid])
    test_sessions = set(session_ids_sorted[n_train + n_valid :])

    return train_sessions, valid_sessions, test_sessions

def augment_sessions(df, session_field, item_field, time_field, max_seq_len):
    """Create sequential augmentation (item_id_list) from raw interactions."""
    df_sorted = df.sort_values([session_field, time_field])
    augmented_rows = []
    
    for session_id, group in tqdm(df_sorted.groupby(session_field), desc="Augmenting sequences"):
        items = group[item_field].tolist()
        times = group[time_field].tolist()
        
        for i in range(1, len(items)):
            target_item = items[i]
            target_time = times[i]
            seq = items[max(0, i - max_seq_len):i]
            
            seq_str = " ".join(map(str, seq))
            
            augmented_rows.append({
                session_field: session_id,
                item_field: target_item,
                "item_id_list": seq_str,
                time_field: target_time
            })
            
    return pd.DataFrame(augmented_rows)

def write_benchmark_inter(df, output_path, session_field, item_field, time_field):
    """Write DataFrame back to .inter format."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{session_field}:token\t{item_field}:token\titem_id_list:token_seq\t{time_field}:float\n")
        
        for _, row in df.iterrows():
            s_id = row[session_field]
            i_id = row[item_field]
            i_list = row["item_id_list"]
            t = row[time_field]
            f.write(f"{s_id}\t{i_id}\t{i_list}\t{t}\n")

def make_benchmark_splits(alias, ratios=[0.8, 0.1, 0.1], max_seq_len=100):
    print(f"\nCreating benchmark splits (Train/Valid/Test) for {alias}...")
    
    inter_path = os.path.join("dataset", alias, f"{alias}.inter")
    if not os.path.isfile(inter_path):
        print(f"File not found: {inter_path}")
        return

    with open(inter_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()

    columns_with_types = header_line.split("\t")
    col_names = [c.split(":")[0] for c in columns_with_types]

    df = pd.read_csv(inter_path, sep="\t", header=0, names=col_names, skiprows=1, dtype=str)

    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(float)

    print(f"  Total interactions: {len(df):,}")
    print(f"  Total sessions:     {df['session_id'].nunique():,}")

    print(f"\n--- Splitting sessions temporally ---")
    train_sessions, valid_sessions, test_sessions = split_sessions_temporal(df, "session_id", "timestamp", ratios)
    print(f"  Train sessions: {len(train_sessions):,}")
    print(f"  Valid sessions: {len(valid_sessions):,}")
    print(f"  Test sessions:  {len(test_sessions):,}")

    train_df = df[df["session_id"].isin(train_sessions)]
    valid_df = df[df["session_id"].isin(valid_sessions)]
    test_df = df[df["session_id"].isin(test_sessions)]

    print(f"\n--- Augmenting sequences (max_seq_len={max_seq_len}) ---")
    train_aug = augment_sessions(train_df, "session_id", "item_id", "timestamp", max_seq_len)
    valid_aug = augment_sessions(valid_df, "session_id", "item_id", "timestamp", max_seq_len)
    test_aug = augment_sessions(test_df, "session_id", "item_id", "timestamp", max_seq_len)

    print(f"  Train augmented rows: {len(train_aug):,}")
    print(f"  Valid augmented rows: {len(valid_aug):,}")
    print(f"  Test augmented rows:  {len(test_aug):,}")

    print(f"\n--- Writing benchmark files ---")
    output_dir = os.path.join("dataset", alias)
    splits = [("train", train_aug), ("valid", valid_aug), ("test", test_aug)]
    
    for split_name, split_df in splits:
        output_path = os.path.join(output_dir, f"{alias}.{split_name}.inter")
        write_benchmark_inter(split_df, output_path, "session_id", "item_id", "timestamp")
        print(f"  {split_name}: {os.path.basename(output_path)} ({len(split_df):,} rows)")

if __name__ == "__main__":
    if not os.path.exists(get_data_file_path(DATA_PATH_RAW, DATA_FILE)):
        initialize()
        filter_by_time_window(365, 65)
        os.remove(get_data_file_path(DATA_PATH_RAW, DATA_FILE))
        safe_copy(
            get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
            get_data_file_path(DATA_PATH_RAW, DATA_FILE)
        )

    if not os.path.exists(get_data_file_path(DATA_PATH_RAW, "tracks")):
        make_track_names_file()

    filter_by_time_window()
    copy_processed_to_temp()
    filter_tracks_by_playcount()
    copy_processed_to_temp()
    make_inter_file(get_dataset_name("30music__"))
    make_tracks_file(get_dataset_name("30music__"))
    make_item_file(get_dataset_name("30music__"))
    make_benchmark_splits(get_dataset_name("30music__"))
    remove_temp_file()