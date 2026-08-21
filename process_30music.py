
import json
import os
import re
import shutil
import stat
import argparse
import subprocess
from collections import Counter
from urllib.parse import unquote_plus
import pandas as pd
from tqdm import tqdm

DATA_FILE = "sessions"
DATA_PATH_RAW = "dataset_raw/"
DATA_PATH_TEMP = "dataset_temp/"
DATA_PATH_PROCESSED = "dataset_processed/"

MIN_TRACK_PLAYCOUNT = 5
MIN_SESSION_LENGTH = 2
MAX_SESSION_LENGTH = 100
MIN_SESSION_PLAYTIME = 30
MAX_SESSION_PLAYTIME = 1_000_000
MAX_SESSION_RECENT_TRACKS = MAX_SESSION_LENGTH
DAYS_FROM_MAX = 165
DAYS_TO_MAX = 65

MAX_VALID_TRACK_ID = 3893303
MIN_TIMESTAMP = 1390209860
MAX_TIMESTAMP = 1421745720

_SESSION_INACTIVITY_GAP = 800
_MIN_TAG_WEIGHT = 0

_ARG_TO_GLOBAL = {
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
    parser = argparse.ArgumentParser(
        description="Process 30Music sessions into RecBole format."
    )
    for arg_name in _ARG_TO_GLOBAL:
        parser.add_argument(f"--{arg_name}", type=int)

    args = parser.parse_args()

    g = globals()
    for arg_name, global_name in _ARG_TO_GLOBAL.items():
        value = getattr(args, arg_name)
        if value is not None:
            g[global_name] = value

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

def _copy_processed_to_temp():
    safe_copy(
        get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
        get_data_file_path(DATA_PATH_TEMP, DATA_FILE),
    )

def _remove_temp_file():
    path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    if os.path.exists(path):
        os.remove(path)

def _iter_session_lines(path, desc="Processing"):
    total = get_line_count(path)
    with open(path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=total, desc=desc):
            parts = line.strip().split("\t")
            tracks = json.loads(parts[3])
            yield parts, tracks

def initialize():
    print("\nInitializing data...")

    input_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE, file_extension=".idomaar")
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

    def _extract_timestamp(line):
        try:
            idx1 = line.find("\t", 14)
            if idx1 == -1:
                return None
            idx2 = line.find("\t", idx1 + 1)
            if idx2 == -1:
                return None
            return int(line[idx1 + 1 : idx2])
        except ValueError:
            return None

    raw_lines = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=get_line_count(input_path), desc=f"Reading {input_path}"):
            timestamp = _extract_timestamp(line)
            if timestamp is not None:
                raw_lines.append((timestamp, line))

    print(f"Sorting {input_path}")
    raw_lines.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for _, line in tqdm(raw_lines, desc=f"Writing {output_path}"):
            line = line[len("event.session\t"):]

            try:
                parts = line.split("\t")
                session_id = int(parts[0])
                session_timestamp = int(parts[1])
                session_stats = json.loads(parts[2][: parts[2].find("} {") + 1])
                session_objects = json.loads(line[line.find("} {") + 2 :].strip())
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

            raw_tracks.sort(key=lambda x: x["ps"])

            sub_sessions = _split_into_sub_sessions(raw_tracks)

            for sub_idx, sub_session in enumerate(sub_sessions):
                if not (MIN_SESSION_LENGTH <= len(sub_session) <= MAX_SESSION_LENGTH):
                    continue

                playtime = sub_session[-1]["ps"] - sub_session[0]["ps"]
                if not (MIN_SESSION_PLAYTIME <= playtime <= MAX_SESSION_PLAYTIME):
                    continue

                new_session_id = (
                    f"{session_id}_{sub_idx}" if len(sub_sessions) > 1 else str(session_id)
                )
                fout.write(
                    f"{new_session_id}\t{session_timestamp}\t{session_user_id}"
                    f"\t{json.dumps(sub_session, separators=(',', ':'))}\n"
                )

    safe_copy(
        get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
        get_data_file_path(DATA_PATH_RAW, DATA_FILE),
    )

def _split_into_sub_sessions(tracks):
    if not tracks:
        return []

    sub_sessions = []
    current = [tracks[0]]
    for prev, cur in zip(tracks, tracks[1:]):
        if cur["ps"] - prev["ps"] > _SESSION_INACTIVITY_GAP:
            sub_sessions.append(current)
            current = []
        current.append(cur)
    sub_sessions.append(current)
    return sub_sessions

def filter_by_time_window(days_from_max=None, days_to_max=None):
    if days_from_max is None:
        days_from_max = DAYS_FROM_MAX
    if days_to_max is None:
        days_to_max = DAYS_TO_MAX
    print(f"\nFiltering sessions: last {days_from_max} to {days_to_max} days from max timestamp...")

    input_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

    lower_bound = MAX_TIMESTAMP - days_from_max * 86400
    upper_bound = MAX_TIMESTAMP - days_to_max * 86400

    kept = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=get_line_count(input_path), desc="Filtering sessions"):
            ts = int(line.split("\t", 3)[1])
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

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=get_line_count(input_path), desc=f"Filtering tracks in {input_path}"):
            parts = line.strip().split("\t")
            session_tracks = json.loads(parts[3])
            session_tracks = [t for t in session_tracks if t["id"] in whitelisted]

            if MIN_SESSION_LENGTH <= len(session_tracks) <= MAX_SESSION_LENGTH:
                fout.write(
                    f"{parts[0]}\t{parts[1]}\t{parts[2]}"
                    f"\t{json.dumps(session_tracks, separators=(',', ':'))}\n"
                )

def make_inter_file(alias):
    print("\nCreating .inter file...")

    input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    output_path = os.path.join("dataset", alias, f"{alias}.inter")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write("session_id:token\tuser_id:token\titem_id:token\ttimestamp:float\n")

        for parts, tracks in _iter_session_lines(input_path, desc=f"Building .inter from {input_path}"):
            session_id = parts[0]
            timestamp = int(parts[1])
            user_id = parts[2]

            for track in tracks:
                fout.write(f"{session_id}\t{user_id}\t{track['id']}\t{timestamp + int(track['ps'])}\n")

def make_tracks_file(alias):
    print("\nCreating tracks file...")

    input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    tracks_source = get_data_file_path(DATA_PATH_RAW, "tracks")
    output_path = os.path.join("dataset", alias, "tracks.tsv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    track_ids = set()
    for _parts, session_tracks in _iter_session_lines(input_path, desc="Collecting track IDs"):
        for t in session_tracks:
            track_ids.add(str(t["id"]))

    seen_tids = set()
    with open(tracks_source, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=get_line_count(tracks_source), desc="Filtering tracks"):
            tid = line.strip().split("\t", 1)[0]
            if tid in track_ids and tid not in seen_tids:
                seen_tids.add(tid)
                fout.write(line)

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
    re.IGNORECASE,
)

_URL_PATTERN = re.compile(
    r'www\.|\.(com|net|org|ru|info)|https?://',
    re.IGNORECASE,
)

def _is_noisy(text):
    return (
        not text
        or text.isspace()
        or '\ufffd' in text
        or sum(1 for c in text if c.isalpha()) < 2
        or _UNKNOWN_PATTERNS.search(text)
        or _URL_PATTERN.search(text)
    )

def make_track_names_file():
    print("\nCreating track names file...")

    sessions_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    tracks_source = os.path.join(DATA_PATH_RAW, "tracks.idomaar")
    output_path = os.path.join(DATA_PATH_RAW, "tracks.tsv")

    track_counts = Counter()
    for _parts, session_tracks in _iter_session_lines(sessions_path, desc="Counting tracks in sessions"):
        for t in session_tracks:
            tid = int(t["id"])
            if tid <= MAX_VALID_TRACK_ID:
                track_counts[tid] += 1

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

def _parse_tags_json(data, min_weight=_MIN_TAG_WEIGHT):
    if not data:
        return []
    if isinstance(data[0], dict):
        return [
            str(t.get("tag", "")).replace(" ", "-")
            for t in data
            if t.get("tag") and int(t.get("weight", 100)) >= min_weight
        ]
    return [str(t).replace(" ", "-") for t in data if t]

def _deduplicate_tags(tags):
    return list(dict.fromkeys(t for t in tags if t))

def _normalize_artist_name(name):
    name = name.strip().lower()
    name = re.sub(r'[\s/\\,;:&+\'"!()\[\]{}]+', '-', name)
    name = re.sub(r'-{2,}', '-', name)
    return name.strip('-') or "unknown-artist"

def _normalize_tag(tag):
    tag = tag.casefold().strip()
    tag = re.sub(r"[\s_]+", "-", tag)
    tag = re.sub(r"[^a-z0-9\-]", "", tag)
    tag = re.sub(r"-+", "-", tag).strip("-")
    return tag

def _load_tag_names(tags_idomaar_path):
    tag_map = {}
    if not os.path.exists(tags_idomaar_path):
        print(f"Warning: {tags_idomaar_path} not found. Track-level tags will be empty.")
        return tag_map

    with open(tags_idomaar_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0] == "tag":
                try:
                    tag_id = int(parts[1])
                    tag_data = json.loads(parts[3])
                    tag_name = tag_data.get("value", "")
                    if tag_name:
                        normalized = _normalize_tag(tag_name)
                        if normalized:
                            tag_map[tag_id] = normalized
                except (json.JSONDecodeError, ValueError):
                    pass
    return tag_map

def _load_track_tags(tracks_idomaar_path, tag_name_map):
    track_tags = {}
    if not os.path.exists(tracks_idomaar_path):
        print(f"Warning: {tracks_idomaar_path} not found. Track-level tags will be empty.")
        return track_tags

    with open(tracks_idomaar_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5 or parts[0] != "track":
                continue
            try:
                track_id = parts[1]
                relationships = json.loads(parts[4])
                tag_entries = relationships.get("tags", [])
                if not tag_entries:
                    continue
                resolved = []
                for t in tag_entries:
                    tid = t.get("id")
                    if tid is not None and tid in tag_name_map:
                        resolved.append(tag_name_map[tid])
                if resolved:
                    track_tags[track_id] = " ".join(_deduplicate_tags(resolved))
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

    return track_tags

def make_item_file(alias):
    print("\nCreating .item file...")

    artist_tags_path = os.path.join("dataset", "artists_tags.tsv")
    tracks_path = os.path.join("dataset", alias, "tracks.tsv")
    output_path = os.path.join("dataset", alias, f"{alias}.item")
    tags_idomaar_path = os.path.join(DATA_PATH_RAW, "tags.idomaar")
    tracks_idomaar_path = os.path.join(DATA_PATH_RAW, "tracks.idomaar")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    artist_tags = _load_artist_tags(artist_tags_path)

    print("Loading track-level tag names from tags.idomaar...")
    tag_name_map = _load_tag_names(tags_idomaar_path)
    print(f"Loaded {len(tag_name_map):,} tag name mappings.")

    print("Loading track-level tags from tracks.idomaar...")
    track_tags_map = _load_track_tags(tracks_idomaar_path, tag_name_map)
    print(f"Loaded track-level tags for {len(track_tags_map):,} tracks.")

    try:
        total_tracks = get_line_count(tracks_path)
    except Exception:
        total_tracks = None

    n_with_track_tags = 0
    n_fallback = 0

    with open(tracks_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        fout.write("item_id:token\tartist_tags:token_seq\ttrack_tags:token_seq\n")

        for line in tqdm(fin, total=total_tracks, desc=f"Building .item from {tracks_path}"):
            parts = line.strip("\n").split("\t")
            if len(parts) < 2:
                continue

            track_id = parts[0]
            artist = parts[1].split("/_/")[0]
            artist_name = _normalize_artist_name(artist)
            a_tags = (artist_tags.get(track_id)
                      or artist_tags.get(artist_name)
                      or "unknown")

            t_tags = track_tags_map.get(track_id)
            if t_tags:
                n_with_track_tags += 1
            else:
                t_tags = a_tags  # fallback
                n_fallback += 1

            fout.write(f"{track_id}\t{a_tags}\n")

    print(f"Track-level tags: {n_with_track_tags:,} tracks with own tags, "
          f"{n_fallback:,} fell back to artist tags.")

def _load_artist_tags(artist_tags_path):
    artist_tags = {}

    if not os.path.exists(artist_tags_path):
        print(f"Warning: {artist_tags_path} not found. All items will have 'unknown' tags.")
        return artist_tags

    with open(artist_tags_path, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.strip("\n").split("\t")
            try:
                if len(parts) >= 4:
                    tags = _parse_tags_json(json.loads(parts[2])) + _parse_tags_json(json.loads(parts[3]))
                    tags = _deduplicate_tags(tags)
                elif len(parts) >= 2:
                    json_col = parts[2] if len(parts) == 3 else parts[1]
                    tags = _parse_tags_json(json.loads(json_col))
                else:
                    continue
                if tags:
                    artist_tags[parts[0]] = " ".join(tags)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                pass

    return artist_tags

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
    session_start_times = df.groupby(session_field)[time_field].min().sort_values()
    session_ids_sorted = session_start_times.index.values

    n_sessions = len(session_ids_sorted)
    n_train = int(n_sessions * ratios[0])
    n_valid = int(n_sessions * ratios[1])

    train_sessions = set(session_ids_sorted[:n_train])
    valid_sessions = set(session_ids_sorted[n_train : n_train + n_valid])
    test_sessions = set(session_ids_sorted[n_train + n_valid :])

    return train_sessions, valid_sessions, test_sessions

def augment_sessions(df, session_field, item_field, time_field, max_seq_len):
    df_sorted = df.sort_values([session_field, time_field])

    sessions = df_sorted[session_field].values
    items = df_sorted[item_field].values.astype(str)
    times = df_sorted[time_field].values

    n = len(sessions)
    if n == 0:
        return pd.DataFrame(columns=[session_field, item_field, "item_id_list", time_field])

    augmented_rows = []
    start_idx = 0

    for i in tqdm(range(1, n), desc="Augmenting sequences"):
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

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)

    output_df.to_csv(output_path, sep="\t", header=False, index=False, mode="a")

def make_benchmark_splits(alias, ratios=None, max_seq_len=100):
    if ratios is None:
        ratios = [0.8, 0.1, 0.1]

    print(f"\nCreating benchmark splits (Train/Valid/Test) for {alias}...")

    inter_path = os.path.join("dataset", alias, f"{alias}.inter")
    if not os.path.isfile(inter_path):
        print(f"File not found: {inter_path}")
        return

    with open(inter_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()

    col_names = [c.split(":")[0] for c in header_line.split("\t")]

    df = pd.read_csv(inter_path, sep="\t", header=0, names=col_names, dtype=str)

    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(float)

    print(f"  Total interactions: {len(df):,}")
    print(f"  Total sessions:     {df['session_id'].nunique():,}")

    print("\n--- Splitting sessions temporally ---")
    split_sets = split_sessions_temporal(df, "session_id", "timestamp", ratios)
    split_names = ("train", "valid", "test")

    for name, sess in zip(split_names, split_sets):
        print(f"  {name.capitalize()} sessions: {len(sess):,}")

    split_dfs = [df[df["session_id"].isin(s)] for s in split_sets]

    print(f"\n--- Augmenting sequences (max_seq_len={max_seq_len}) ---")
    output_dir = os.path.join("dataset", alias)

    for name, split_df in zip(split_names, split_dfs):
        aug = augment_sessions(split_df, "session_id", "item_id", "timestamp", max_seq_len)
        print(f"  {name.capitalize()} augmented rows: {len(aug):,}")
        output_path = os.path.join(output_dir, f"{alias}.{name}.inter")
        write_benchmark_inter(aug, output_path, "session_id", "item_id", "timestamp")
        print(f"  Written: {os.path.basename(output_path)}")

def main():
    parse_args()

    raw_sessions = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    if not os.path.exists(raw_sessions):
        initialize()
        filter_by_time_window(165, 65)
        os.remove(raw_sessions)
        safe_copy(get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE), raw_sessions)

    if not os.path.exists(get_data_file_path(DATA_PATH_RAW, "tracks")):
        make_track_names_file()

    dataset_name = get_dataset_name("30music__")

    filter_by_time_window()
    _copy_processed_to_temp()
    filter_tracks_by_playcount()
    _copy_processed_to_temp()
    make_inter_file(dataset_name)
    make_tracks_file(dataset_name)
    make_item_file(dataset_name)
    make_benchmark_splits(dataset_name)
    _remove_temp_file()

if __name__ == "__main__":
    main()

