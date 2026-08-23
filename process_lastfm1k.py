
import json
import os
import re
import shutil
import stat
import argparse
import subprocess
from collections import Counter
from datetime import datetime, timezone
import pandas as pd
from tqdm import tqdm
import uuid

DATA_FILE = "lastfm_sessions"
DATA_PATH_RAW = "dataset_raw/"
DATA_PATH_TEMP = "dataset_temp/"
DATA_PATH_PROCESSED = "dataset_processed/"

LASTFM_RAW_FILE = os.path.join(DATA_PATH_RAW, "userid-timestamp-artid-artname-traid-traname.tsv")
LASTFM_RAW_TRACKS_FILE = os.path.join(DATA_PATH_RAW, "lastfm_tracks_raw.tsv")
LASTFM_TRACKS_FILE = os.path.join(DATA_PATH_RAW, "lastfm_tracks.tsv")

MIN_TRACK_PLAYCOUNT = 5
MIN_SESSION_LENGTH = 2
MAX_SESSION_LENGTH = 100
MIN_SESSION_PLAYTIME = 30
MAX_SESSION_PLAYTIME = 1_000_000
MAX_SESSION_RECENT_TRACKS = MAX_SESSION_LENGTH
DAYS_FROM_MAX = 366
DAYS_TO_MAX = 0

START_TIMESTAMP = 1199145600
END_TIMESTAMP = 1230767999

SESSION_INACTIVITY_GAP = 800
_MIN_TAG_WEIGHT = 0

ARG_TO_GLOBAL = {
    "min_track_playcount": "MIN_TRACK_PLAYCOUNT",
    "min_session_length": "MIN_SESSION_LENGTH",
    "max_session_length": "MAX_SESSION_LENGTH",
    "min_session_playtime": "MIN_SESSION_PLAYTIME",
    "max_session_playtime": "MAX_SESSION_PLAYTIME",
    "max_session_recent_tracks": "MAX_SESSION_RECENT_TRACKS",
    "days_from_max": "DAYS_FROM_MAX",
    "days_to_max": "DAYS_TO_MAX",
}

def parse_args():
    parser = argparse.ArgumentParser()

    for arg_name in ARG_TO_GLOBAL:
        parser.add_argument(f"--{arg_name}", type=int)

    args = parser.parse_args()
    global_vars = globals()

    for arg_name, global_name in ARG_TO_GLOBAL.items():
        value = getattr(args, arg_name)
        
        if value is not None:
            global_vars[global_name] = value

    return args

def get_data_file_path(data_path, data_file, file_extension=".tsv"):
    return os.path.join(data_path, f"{data_file}{file_extension}")

def get_line_count(file_path):
    return int(subprocess.check_output(["wc", "-l", file_path]).split()[0])

def safe_copy(src, dst):
    if os.path.exists(dst):
        try:
            os.chmod(dst, stat.S_IWRITE)
            os.remove(dst)
        except OSError:
            pass
    shutil.copyfile(src, dst)

def copy_processed_to_temp():
    safe_copy(get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE), get_data_file_path(DATA_PATH_TEMP, DATA_FILE))

def get_sessions(path):
    with open(path, "r", encoding="utf-8") as file_in:
        for line in tqdm(file_in, total=get_line_count(path), desc="Processing"):
            parts = line.strip().split("\t")
            tracks = json.loads(parts[3])
            yield parts, tracks

_NOISE_PATTERNS = re.compile(
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
    r'|various\s*artists?'
    r'www\.|\.(com|net|org|ru|info)|https?://',
    re.IGNORECASE,
)

def is_noisy(text):
    return (
        not text
        or text.isspace()
        or '\ufffd' in text
        or sum(1 for c in text if c.isalpha()) < 2
        or _NOISE_PATTERNS.search(text)
    )

def parse_timestamp(ts_str):
    try:
        dt = datetime(
            int(ts_str[:4]), int(ts_str[5:7]), int(ts_str[8:10]),
            int(ts_str[11:13]), int(ts_str[14:16]), int(ts_str[17:19]),
            tzinfo=timezone.utc
        )
        return int(dt.timestamp())
    except Exception:
        return None

def get_dataset_name(prefix="lastfm1k__"):
    name_parts = [
        f"days[{DAYS_FROM_MAX}-{DAYS_TO_MAX}]",
        f"pcount[{MIN_TRACK_PLAYCOUNT}]",
        f"ptime[{MIN_SESSION_PLAYTIME}-{MAX_SESSION_PLAYTIME}]",
        f"length[{MIN_SESSION_LENGTH}-{MAX_SESSION_LENGTH}]",
        f"recent[{MAX_SESSION_RECENT_TRACKS}]",
    ]
    return prefix + "_".join(name_parts)

def initialize():
    print("\nInitializing")

    input_path = LASTFM_RAW_FILE
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

    user_scrobbles = {}

    with open(input_path, "r", encoding="utf-8", errors="replace") as file_in:
        for line in tqdm(file_in, total=get_line_count(input_path), desc=f"Reading {input_path}"):
            parts = line.strip().split("\t")

            if len(parts) < 6:
                continue

            user_id = parts[0].strip()
            timestamp = parse_timestamp(parts[1].strip())
            artist_name = parts[3].strip()
            track_name = parts[5].strip()

            if not user_id or is_noisy(artist_name) or is_noisy(track_name):
                continue

            if timestamp is None:
                continue

            if not (START_TIMESTAMP <= timestamp <= END_TIMESTAMP):
                continue

            track_key = f"{artist_name}/_/{track_name}"

            user_scrobbles.setdefault(user_id, []).append((timestamp, track_key))

    track_key_to_id = {}
    for scrobbles in user_scrobbles.values():
        for _, track_key in scrobbles:
            if track_key not in track_key_to_id:
                track_key_to_id[track_key] = len(track_key_to_id) + 1

    with open(LASTFM_RAW_TRACKS_FILE, "w", encoding="utf-8") as file_out:
        for track_key, track_id in sorted(track_key_to_id.items(), key=lambda x: x[1]):
            file_out.write(f"{track_id}\t{track_key}\n")

    session_counter = 0

    with open(output_path, "w", encoding="utf-8") as file_out:
        for user_id, scrobbles in tqdm(user_scrobbles.items(), desc="Generating sessions"):
            scrobbles.sort(key=lambda x: x[0])

            deduplicated_tracks = []
            timestamps = {}
            for timestamp, track_key in scrobbles:
                track_id = track_key_to_id[track_key]
                if track_id in timestamps and abs(timestamp - timestamps[track_id]) < 10:
                    continue
                timestamps[track_id] = timestamp
                deduplicated_tracks.append({"id": track_id, "ps": timestamp})

            if not deduplicated_tracks:
                continue

            sub_sessions = make_sub_sessions(deduplicated_tracks)

            for sub_idx, sub_session in enumerate(sub_sessions):
                if not (MIN_SESSION_LENGTH <= len(sub_session) <= MAX_SESSION_LENGTH):
                    continue

                playtime = sub_session[-1]["ps"] - sub_session[0]["ps"]
                if not (MIN_SESSION_PLAYTIME <= playtime <= MAX_SESSION_PLAYTIME):
                    continue

                session_counter += 1
                session_timestamp = sub_session[0]["ps"]
                new_session_id = f"{session_counter}_{sub_idx}"

                rel_sub_session = [
                    {"id": track["id"], "ps": track["ps"] - session_timestamp}
                    for track in sub_session
                ]

                file_out.write(
                    f"{new_session_id}\t{session_timestamp}\t{user_id}"
                    f"\t{json.dumps(rel_sub_session, separators=(',', ':'))}\n"
                )

    safe_copy(get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE), get_data_file_path(DATA_PATH_RAW, DATA_FILE))

def make_sub_sessions(tracks):
    if not tracks:
        return []

    sub_sessions = []
    current = [tracks[0]]

    for previous_track, current_track in zip(tracks, tracks[1:]):
        if current_track["ps"] - previous_track["ps"] > SESSION_INACTIVITY_GAP:
            sub_sessions.append(current)
            current = []

        current.append(current_track)

    sub_sessions.append(current)

    return sub_sessions

def get_min_max_timestamps():
    input_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    min_timestamp = float("inf")
    max_timestamp = 0
    with open(input_path, "r", encoding="utf-8") as file_in:
        for line in file_in:
            parts = line.split("\t", 2)
            if len(parts) >= 2:
                try:
                    timestamp = int(parts[1])
                    if timestamp < min_timestamp:
                        min_timestamp = timestamp
                    if timestamp > max_timestamp:
                        max_timestamp = timestamp
                except ValueError:
                    pass
    return int(min_timestamp), int(max_timestamp)

def get_max_timestamp():
    _, max_ts = get_min_max_timestamps()
    return max_ts

def get_5_equal_splits():
    min_ts, max_ts = get_min_max_timestamps()
    total_span = max_ts - min_ts
    step = total_span / 5.0

    splits = []
    for k in range(5):
        start_ts = min_ts + k * step
        end_ts = min_ts + (k + 1) * step if k < 4 else max_ts
        days_from_max = int(round((max_ts - start_ts) / 86400.0))
        days_to_max = int(round((max_ts - end_ts) / 86400.0))
        splits.append((days_from_max, days_to_max))
    return splits

def filter_by_time_window(days_from_max=None, days_to_max=None, max_timestamp=None):
    if days_from_max is None:
        days_from_max = DAYS_FROM_MAX

    if days_to_max is None:
        days_to_max = DAYS_TO_MAX

    if max_timestamp is None:
        max_timestamp = get_max_timestamp()

    print(f"\nFiltering sessions last {days_from_max} to {days_to_max} days")

    input_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

    lower_bound = max_timestamp - days_from_max * 86400
    upper_bound = max_timestamp - days_to_max * 86400

    with open(input_path, "r", encoding="utf-8") as file_in, open(output_path, "w", encoding="utf-8") as file_out:
        for line in tqdm(file_in, total=get_line_count(input_path), desc="Filtering sessions"):
            timestamp = int(line.split("\t", 3)[1])

            if lower_bound <= timestamp <= upper_bound:
                file_out.write(line)

def filter_tracks_by_playcount():
    print("\nFiltering tracks by whitelist")

    input_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    tracks_file = LASTFM_TRACKS_FILE

    allowed_tracks = set()
    with open(tracks_file, "r", encoding="utf-8") as file_in:
        for line in file_in:
            allowed_tracks.add(int(line.strip().split("\t", 1)[0]))

    with open(input_path, "r", encoding="utf-8") as file_in, open(output_path, "w", encoding="utf-8") as file_out:
        for line in tqdm(file_in, total=get_line_count(input_path), desc=f"Filtering tracks in {input_path}"):
            parts = line.strip().split("\t")
            session_tracks = json.loads(parts[3])
            session_tracks = [track for track in session_tracks if track["id"] in allowed_tracks]

            if MIN_SESSION_LENGTH <= len(session_tracks) <= MAX_SESSION_LENGTH:
                file_out.write(
                    f"{parts[0]}\t{parts[1]}\t{parts[2]}"
                    f"\t{json.dumps(session_tracks, separators=(',', ':'))}\n"
                )

def make_inter_file(alias):
    print("\nCreating .inter file")

    input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    output_path = os.path.join("dataset", alias, f"{alias}.inter")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file_out:
        file_out.write("session_id:token\tuser_id:token\titem_id:token\ttimestamp:float\n")

        for parts, tracks in get_sessions(input_path):
            session_id = parts[0]
            timestamp = int(parts[1])
            user_id = parts[2]

            for track in tracks:
                file_out.write(f"{session_id}\t{user_id}\t{track['id']}\t{timestamp + int(track['ps'])}\n")

def make_tracks_file(alias):
    print("\nCreating tracks file")

    input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    tracks_source = LASTFM_TRACKS_FILE
    output_path = os.path.join("dataset", alias, "tracks.tsv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    track_ids = set()
    for _, session_tracks in get_sessions(input_path):
        for track in session_tracks:
            track_ids.add(str(track["id"]))

    written_tracks = set()
    with open(tracks_source, "r", encoding="utf-8") as file_in, open(output_path, "w", encoding="utf-8") as file_out:
        for line in tqdm(file_in, total=get_line_count(tracks_source), desc="Filtering tracks"):
            track_id = line.strip().split("\t", 1)[0]

            if track_id in track_ids and track_id not in written_tracks:
                written_tracks.add(track_id)
                file_out.write(line)

def make_track_names_file():
    print("\nCreating track names file")

    sessions_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    tracks_source = LASTFM_RAW_TRACKS_FILE
    output_path = LASTFM_TRACKS_FILE

    track_counts = Counter()
    for _, session_tracks in get_sessions(sessions_path):
        for track in session_tracks:
            track_counts[int(track["id"])] += 1

    track_ids = {track_id for track_id, count in track_counts.items() if count >= MIN_TRACK_PLAYCOUNT}

    tracks = {}
    with open(tracks_source, "r", encoding="utf-8") as file_in:
        for line in tqdm(file_in, total=get_line_count(tracks_source), desc="Loading track names"):
            parts = line.strip().split("\t", 1)

            if len(parts) < 2:
                continue

            track_id = int(parts[0])

            if track_id in track_ids:
                tracks[track_id] = parts[1]

    with open(output_path, "w", encoding="utf-8") as file_out:
        for track_id, name in sorted(tracks.items()):
            file_out.write(f"{track_id}\t{name}\n")

def parse_tags_json(tags, min_weight=_MIN_TAG_WEIGHT):
    if not tags:
        return []
    if isinstance(tags[0], dict):
        return [
            str(tag.get("tag", "")).replace(" ", "-")
            for tag in tags
            if tag.get("tag") and int(tag.get("weight", 100)) >= min_weight
        ]
    return [str(tag).replace(" ", "-") for tag in tags if tag]

def normalize_artist_name(name):
    name = name.strip().lower()
    name = re.sub(r'[\s/\\,;:&+\'"!()\[\]{}]+', '-', name)
    name = re.sub(r'-{2,}', '-', name)
    return name.strip('-') or "unknown-artist"

def get_artist_tags(artist_tags_path):
    artist_tags = {}

    with open(artist_tags_path, "r", encoding="utf-8") as file_in:
        for line in file_in:
            parts = line.strip("\n").split("\t")
            try:
                if len(parts) >= 4:
                    tags = parse_tags_json(json.loads(parts[2])) + parse_tags_json(json.loads(parts[3]))
                    tags = list(dict.fromkeys(tag for tag in tags if tag))
                elif len(parts) >= 2:
                    json_col = parts[2] if len(parts) == 3 else parts[1]
                    tags = parse_tags_json(json.loads(json_col))
                else:
                    continue
                if tags:
                    artist_tags[parts[0]] = " ".join(tags)
                    if len(parts) >= 2:
                        norm_artist = normalize_artist_name(parts[1])
                        if norm_artist:
                            artist_tags[norm_artist] = " ".join(tags)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                pass

    return artist_tags

def make_item_file(alias):
    print("\nCreating .item file")

    artist_tags_path = os.path.join("dataset", "artists_tags_all.tsv")
    tracks_path = os.path.join("dataset", alias, "tracks.tsv")
    output_path = os.path.join("dataset", alias, f"{alias}.item")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_artist_tags = get_artist_tags(artist_tags_path)

    with open(tracks_path, "r", encoding="utf-8") as file_in, open(output_path, "w", encoding="utf-8") as file_out:
        file_out.write("item_id:token\tartist_tags:token_seq\n")

        for line in tqdm(file_in, total=get_line_count(tracks_path), desc=f"Building .item from {tracks_path}"):
            parts = line.strip("\n").split("\t")

            if len(parts) < 2:
                continue

            track_id = parts[0]
            artist = parts[1].split("/_/")[0]
            artist_name = normalize_artist_name(artist)
            artist_tags = (all_artist_tags.get(track_id) or all_artist_tags.get(artist_name) or str(uuid.uuid4()))

            file_out.write(f"{track_id}\t{artist_tags}\n")

def train_valid_test_split(df, session_field, time_field, ratios):
    session_start_times = df.groupby(session_field)[time_field].min().sort_values()
    session_ids_sorted = session_start_times.index.values

    num_sessions = len(session_ids_sorted)
    num_train = int(num_sessions * ratios[0])
    num_valid = int(num_sessions * ratios[1])

    train_sessions = set(session_ids_sorted[:num_train])
    valid_sessions = set(session_ids_sorted[num_train : num_train + num_valid])
    test_sessions = set(session_ids_sorted[num_train + num_valid :])

    return train_sessions, valid_sessions, test_sessions

def augment_sessions(df, session_field, item_field, time_field, max_seq_len):
    df_sorted = df.sort_values([session_field, time_field])

    sessions = df_sorted[session_field].values
    items = df_sorted[item_field].values.astype(str)
    times = df_sorted[time_field].values

    augmented_rows = []
    start_idx = 0

    for i in tqdm(range(1, len(sessions)), desc="Augmenting sequences"):
        if sessions[i] != sessions[i - 1]:
            start_idx = i
            continue

        seq_start = max(start_idx, i - max_seq_len)
        augmented_rows.append({
            session_field: sessions[i],
            item_field: items[i],
            "item_id_list": " ".join(items[seq_start:i]),
            time_field: times[i],
        })

    return pd.DataFrame(augmented_rows)

def write_benchmark_inter(df, output_path, session_field, item_field, time_field):
    header = f"{session_field}:token\t{item_field}:token\titem_id_list:token_seq\t{time_field}:float\n"

    output_df = df[[session_field, item_field, "item_id_list", time_field]]

    with open(output_path, "w", encoding="utf-8") as file_out:
        file_out.write(header)

    output_df.to_csv(output_path, sep="\t", header=False, index=False, mode="a")

def make_benchmark_splits(alias):
    print(f"\nCreating benchmark splits")

    inter_path = os.path.join("dataset", alias, f"{alias}.inter")

    with open(inter_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()

    col_names = [column.split(":")[0] for column in header_line.split("\t")]
    df = pd.read_csv(inter_path, sep="\t", header=0, names=col_names, dtype=str)

    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(float)

    split_sets = train_valid_test_split(df, "session_id", "timestamp", [0.8, 0.1, 0.1])
    split_names = ("train", "valid", "test")
    split_dfs = [df[df["session_id"].isin(s)] for s in split_sets]

    output_dir = os.path.join("dataset", alias)

    for name, split_df in zip(split_names, split_dfs):
        aug = augment_sessions(split_df, "session_id", "item_id", "timestamp", 100)
        output_path = os.path.join(output_dir, f"{alias}.{name}.inter")
        write_benchmark_inter(aug, output_path, "session_id", "item_id", "timestamp")

if __name__ == "__main__":
    parse_args()

    initialize()
    make_track_names_file()

    _, max_timestamp = get_min_max_timestamps()

    splits = get_5_equal_splits()
    for day_from, day_to in splits:
        DAYS_FROM_MAX = day_from
        DAYS_TO_MAX = day_to
        dataset_name = get_dataset_name()

        filter_by_time_window(days_from_max=day_from, days_to_max=day_to, max_timestamp=max_timestamp)
        copy_processed_to_temp()
        filter_tracks_by_playcount()
        copy_processed_to_temp()
        make_inter_file(dataset_name)
        make_tracks_file(dataset_name)
        make_item_file(dataset_name)
        make_benchmark_splits(dataset_name)

        path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
        if os.path.exists(path):
            os.remove(path)