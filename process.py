"""Pipeline for processing 30Music session data into RecBole format.

Reads raw .idomaar session files, filters by time window and track playcount,
then produces .inter and .item files suitable for RecBole benchmarking.
"""

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

# ---------------------------------------------------------------------------
# Default constants (overridable via CLI arguments)
# ---------------------------------------------------------------------------

DATA_FILE = "sessions"
DATA_PATH_RAW = "dataset_raw/"
DATA_PATH_TEMP = "dataset_temp/"
DATA_PATH_PROCESSED = "dataset_processed/"

MIN_TRACK_PLAYCOUNT = 25
MIN_SESSION_LENGTH = 2
MAX_SESSION_LENGTH = 100
MIN_SESSION_PLAYTIME = 30
MAX_SESSION_PLAYTIME = 1_000_000
MAX_SESSION_RECENT_TRACKS = MAX_SESSION_LENGTH
DAYS_FROM_MAX = 365
DAYS_TO_MAX = 65

MAX_VALID_TRACK_ID = 3893303
MIN_TIMESTAMP = 1390209860
MAX_TIMESTAMP = 1421745720

# Inactivity gap (seconds) used to split sessions in initialize()
_SESSION_INACTIVITY_GAP = 1800

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

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
    """Parse CLI arguments and override corresponding module-level constants."""
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


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------

def get_data_file_path(data_path, data_file, file_extension=".tsv"):
    """Return the full path for a data file with the given extension."""
    return os.path.join(data_path, f"{data_file}{file_extension}")


def get_line_count(file_path):
    """Return the number of lines in *file_path* using ``wc -l``."""
    return int(subprocess.check_output(["wc", "-l", file_path]).split()[0])


def safe_copy(src, dst):
    """Copy *src* to *dst*, removing a read-only *dst* first if necessary."""
    if os.path.exists(dst):
        try:
            os.chmod(dst, stat.S_IWRITE)
            os.remove(dst)
        except OSError:
            pass
    shutil.copyfile(src, dst)


def copy_processed_to_temp():
    """Copy the processed sessions file into the temp directory."""
    safe_copy(
        get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
        get_data_file_path(DATA_PATH_TEMP, DATA_FILE),
    )


def remove_temp_file():
    """Remove the temporary sessions file if it exists."""
    temp_file_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)


# ---------------------------------------------------------------------------
# Session-line iterator helper (eliminates repeated boilerplate)
# ---------------------------------------------------------------------------

def _iter_session_lines(path, desc="Processing"):
    """Yield ``(parts, tracks)`` for each line in a sessions TSV file.

    *parts* is the tab-split list of raw fields; *tracks* is the parsed JSON
    track list from ``parts[3]``.
    """
    total = get_line_count(path)
    with open(path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=total, desc=desc):
            parts = line.strip().split("\t")
            tracks = json.loads(parts[3])
            yield parts, tracks


# ---------------------------------------------------------------------------
# Initialization (raw .idomaar → sorted TSV)
# ---------------------------------------------------------------------------

def initialize():
    """Parse the raw ``.idomaar`` file, sort by timestamp, split into
    sub-sessions, and write the initial processed TSV."""
    print("\nInitializing data...")

    input_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE, file_extension=".idomaar")
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

    def _extract_timestamp(line):
        """Extract the timestamp field from a raw idomaar line."""
        try:
            # "event.session\t" is 14 chars
            idx1 = line.find("\t", 14)
            if idx1 == -1:
                return None
            idx2 = line.find("\t", idx1 + 1)
            if idx2 == -1:
                return None
            return int(line[idx1 + 1 : idx2])
        except ValueError:
            return None

    # -- Read & sort --------------------------------------------------------
    raw_lines = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=get_line_count(input_path), desc=f"Reading {input_path}"):
            timestamp = _extract_timestamp(line)
            if timestamp is not None:
                raw_lines.append((timestamp, line))

    print(f"Sorting {input_path}")
    raw_lines.sort(key=lambda x: x[0])

    # -- Write sorted sessions ---------------------------------------------
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

            # De-duplicate near-simultaneous plays of the same track
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

            # Split into sub-sessions on inactivity gaps
            sub_sessions = _split_into_sub_sessions(raw_tracks)

            # Write out valid sub-sessions
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
    """Split a chronologically sorted track list into sub-sessions.

    A new sub-session starts when the gap between consecutive play-starts
    exceeds ``_SESSION_INACTIVITY_GAP`` seconds.
    """
    if not tracks:
        return []

    sub_sessions = []
    current_sub = [tracks[0]]

    for i in range(1, len(tracks)):
        if tracks[i]["ps"] - current_sub[-1]["ps"] > _SESSION_INACTIVITY_GAP:
            sub_sessions.append(current_sub)
            current_sub = [tracks[i]]
        else:
            current_sub.append(tracks[i])
    sub_sessions.append(current_sub)

    return sub_sessions


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_by_time_window(days_from_max=None, days_to_max=None):
    """Keep only sessions whose timestamp falls within
    ``[MAX_TIMESTAMP - days_from_max*86400, MAX_TIMESTAMP - days_to_max*86400]``.
    """
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
            parts = line.strip().split("\t")
            ts = int(parts[1])
            if lower_bound <= ts <= upper_bound:
                fout.write(line)
                kept += 1

    print(f"Kept {kept:,} sessions")


def filter_tracks_by_playcount():
    """Remove tracks not present in the ``tracks.tsv`` whitelist, then
    discard sessions that fall outside the allowed length range."""
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


# ---------------------------------------------------------------------------
# Output file builders
# ---------------------------------------------------------------------------

def make_inter_file(alias):
    """Create the ``.inter`` interactions file for RecBole."""
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
    """Create a filtered ``tracks.tsv`` containing only tracks that appear
    in the processed sessions."""
    print("\nCreating tracks file...")

    input_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    tracks_source = get_data_file_path(DATA_PATH_RAW, "tracks")
    output_path = os.path.join("dataset", alias, "tracks.tsv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect all track IDs referenced in processed sessions
    track_ids = set()
    for _parts, session_tracks in _iter_session_lines(input_path, desc="Collecting track IDs"):
        for t in session_tracks:
            track_ids.add(str(t["id"]))

    # Write only matching tracks (deduplicate by track ID)
    seen_tids = set()
    with open(tracks_source, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=get_line_count(tracks_source), desc="Filtering tracks"):
            tid = line.strip().split("\t", 1)[0]
            if tid in track_ids and tid not in seen_tids:
                seen_tids.add(tid)
                fout.write(line)


# ---------------------------------------------------------------------------
# Noise detection for track names
# ---------------------------------------------------------------------------

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
    """Return True if *text* looks like a placeholder, URL, or garbage."""
    if not text or text.isspace():
        return True
    if '\ufffd' in text:
        return True
    if sum(1 for c in text if c.isalpha()) < 2:
        return True
    if _UNKNOWN_PATTERNS.search(text):
        return True
    if _URL_PATTERN.search(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Track-names file (from raw .idomaar)
# ---------------------------------------------------------------------------

def make_track_names_file():
    """Build ``tracks.tsv`` in the raw directory from the ``.idomaar`` source,
    keeping only tracks with at least ``MIN_TRACK_PLAYCOUNT`` plays and
    non-noisy names."""
    print("\nCreating track names file...")

    sessions_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    tracks_source = os.path.join(DATA_PATH_RAW, "tracks.idomaar")
    output_path = os.path.join(DATA_PATH_RAW, "tracks.tsv")

    # Count track occurrences across all sessions
    track_counts = Counter()
    for _parts, session_tracks in _iter_session_lines(sessions_path, desc="Counting tracks in sessions"):
        for t in session_tracks:
            tid = int(t["id"])
            if tid <= MAX_VALID_TRACK_ID:
                track_counts[tid] += 1

    track_ids = {tid for tid, count in track_counts.items() if count >= MIN_TRACK_PLAYCOUNT}
    print(f"Found {len(track_counts):,} unique tracks, {len(track_ids):,} with >= {MIN_TRACK_PLAYCOUNT} plays")

    # Load track names from the idomaar source
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


# ---------------------------------------------------------------------------
# Tag parsing helpers (used by make_item_file)
# ---------------------------------------------------------------------------

def _parse_tags_json(data):
    """Extract a flat list of hyphenated tag strings from a JSON-parsed
    tag structure.

    Handles both ``[{"tag": "rock"}, ...]`` and ``["rock", ...]`` formats.
    """
    if not data:
        return []
    if isinstance(data[0], dict):
        return [
            str(t.get("tag", "")).replace(" ", "-")
            for t in data
            if t.get("tag")
        ]
    return [str(t).replace(" ", "-") for t in data if t]


def _deduplicate_tags(tags):
    """Return *tags* with duplicates removed, preserving insertion order."""
    return list(dict.fromkeys(t for t in tags if t))


# ---------------------------------------------------------------------------
# .item file builder
# ---------------------------------------------------------------------------

def make_item_file(alias):
    """Create the ``.item`` file mapping track IDs to artist tags."""
    print("\nCreating .item file...")

    artist_tags_path = os.path.join("dataset", "artists_tags.tsv")
    tracks_path = os.path.join("dataset", alias, "tracks.tsv")
    output_path = os.path.join("dataset", alias, f"{alias}.item")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    artist_tags = _load_artist_tags(artist_tags_path)

    try:
        total_tracks = get_line_count(tracks_path)
    except Exception:
        total_tracks = None

    with open(tracks_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        fout.write("item_id:token\titem_tags:token_seq\n")

        for line in tqdm(fin, total=total_tracks, desc=f"Building .item from {tracks_path}"):
            parts = line.strip("\n").split("\t")
            if len(parts) < 2:
                continue

            track_id = parts[0]
            track_info = parts[1]

            artist_name = track_info.split("/_/")[0]

            tags = artist_tags.get(track_id)
            if tags is None:
                tags = artist_tags.get(artist_name, "")

            if not tags:
                tags = "unknown"

            fout.write(f"{track_id}\t{tags}\n")


def _load_artist_tags(artist_tags_path):
    """Load the artist-tags mapping from ``artists_tags.tsv``.

    Returns a ``dict[str, str]`` mapping track/artist IDs to space-joined
    tag strings.  Handles 2-, 3-, and 4-column formats.
    """
    artist_tags = {}

    if not os.path.exists(artist_tags_path):
        print(f"Warning: {artist_tags_path} not found. All items will have 'unknown' tags.")
        return artist_tags

    with open(artist_tags_path, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.strip("\n").split("\t")
            try:
                if len(parts) >= 4:
                    _parse_4col_tags(parts, artist_tags)
                elif len(parts) == 3:
                    _parse_simple_tags(parts[0], parts[2], artist_tags)
                elif len(parts) == 2:
                    _parse_simple_tags(parts[0], parts[1], artist_tags)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                pass

    return artist_tags


def _parse_4col_tags(parts, artist_tags):
    """Parse a 4-column artist-tags line (track_id, ?, lastfm_json, mb_json)."""
    track_id = parts[0]
    lastfm_json = json.loads(parts[2])
    mb_json = json.loads(parts[3])

    tags = _parse_tags_json(lastfm_json) + _parse_tags_json(mb_json)
    unique_tags = _deduplicate_tags(tags)

    if unique_tags:
        artist_tags[track_id] = " ".join(unique_tags)


def _parse_simple_tags(key, json_str, artist_tags):
    """Parse a 2- or 3-column artist-tags line."""
    tags_json = json.loads(json_str)
    tags = _parse_tags_json(tags_json)

    if tags:
        artist_tags[key] = " ".join(tags)


# ---------------------------------------------------------------------------
# Dataset naming
# ---------------------------------------------------------------------------

def get_dataset_name(prefix="30music__"):
    """Build a descriptive dataset name encoding the current filter settings."""
    name_parts = [
        f"days[{DAYS_FROM_MAX}-{DAYS_TO_MAX}]",
        f"pcount[{MIN_TRACK_PLAYCOUNT}]",
        f"ptime[{MIN_SESSION_PLAYTIME}-{MAX_SESSION_PLAYTIME}]",
        f"length[{MIN_SESSION_LENGTH}-{MAX_SESSION_LENGTH}]",
        f"recent[{MAX_SESSION_RECENT_TRACKS}]",
    ]
    return prefix + "_".join(name_parts)


# ---------------------------------------------------------------------------
# Benchmark splits (train / valid / test)
# ---------------------------------------------------------------------------

def split_sessions_temporal(df, session_field, time_field, ratios):
    """Split entire sessions by temporal order of their earliest timestamp."""
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
    """Create sequential augmentation (``item_id_list``) from raw interactions.

    For each interaction (except the first in a session), produces one row
    containing the preceding item sequence and the current item as target.
    """
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
    """Write an augmented DataFrame to ``.inter`` format.

    Uses vectorised ``to_csv`` instead of row-by-row ``iterrows``.
    """
    header = f"{session_field}:token\t{item_field}:token\titem_id_list:token_seq\t{time_field}:float\n"

    output_df = df[[session_field, item_field, "item_id_list", time_field]]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)

    output_df.to_csv(output_path, sep="\t", header=False, index=False, mode="a")


def make_benchmark_splits(alias, ratios=None, max_seq_len=100):
    """Split the ``.inter`` file into train/valid/test, augment sequences,
    and write the benchmark files."""
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

    df = pd.read_csv(inter_path, sep="\t", header=0, names=col_names, skiprows=1, dtype=str)

    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(float)

    print(f"  Total interactions: {len(df):,}")
    print(f"  Total sessions:     {df['session_id'].nunique():,}")

    # -- Temporal split -----------------------------------------------------
    print("\n--- Splitting sessions temporally ---")
    train_sessions, valid_sessions, test_sessions = split_sessions_temporal(
        df, "session_id", "timestamp", ratios
    )
    print(f"  Train sessions: {len(train_sessions):,}")
    print(f"  Valid sessions: {len(valid_sessions):,}")
    print(f"  Test sessions:  {len(test_sessions):,}")

    train_df = df[df["session_id"].isin(train_sessions)]
    valid_df = df[df["session_id"].isin(valid_sessions)]
    test_df = df[df["session_id"].isin(test_sessions)]

    # -- Augmentation -------------------------------------------------------
    print(f"\n--- Augmenting sequences (max_seq_len={max_seq_len}) ---")
    train_aug = augment_sessions(train_df, "session_id", "item_id", "timestamp", max_seq_len)
    valid_aug = augment_sessions(valid_df, "session_id", "item_id", "timestamp", max_seq_len)
    test_aug = augment_sessions(test_df, "session_id", "item_id", "timestamp", max_seq_len)

    print(f"  Train augmented rows: {len(train_aug):,}")
    print(f"  Valid augmented rows: {len(valid_aug):,}")
    print(f"  Test augmented rows:  {len(test_aug):,}")

    # -- Write files --------------------------------------------------------
    print("\n--- Writing benchmark files ---")
    output_dir = os.path.join("dataset", alias)
    splits = [("train", train_aug), ("valid", valid_aug), ("test", test_aug)]

    for split_name, split_df in splits:
        output_path = os.path.join(output_dir, f"{alias}.{split_name}.inter")
        write_benchmark_inter(split_df, output_path, "session_id", "item_id", "timestamp")
        print(f"  {split_name}: {os.path.basename(output_path)} ({len(split_df):,} rows)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run the full processing pipeline."""
    parse_args()

    if not os.path.exists(get_data_file_path(DATA_PATH_RAW, DATA_FILE)):
        initialize()
        filter_by_time_window(365, 65)
        os.remove(get_data_file_path(DATA_PATH_RAW, DATA_FILE))
        safe_copy(
            get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
            get_data_file_path(DATA_PATH_RAW, DATA_FILE),
        )

    if not os.path.exists(get_data_file_path(DATA_PATH_RAW, "tracks")):
        make_track_names_file()

    dataset_name = get_dataset_name("30music__")

    filter_by_time_window()
    copy_processed_to_temp()
    filter_tracks_by_playcount()
    copy_processed_to_temp()
    make_inter_file(dataset_name)
    make_tracks_file(dataset_name)
    make_item_file(dataset_name)
    make_benchmark_splits(dataset_name)
    remove_temp_file()


if __name__ == "__main__":
    main()