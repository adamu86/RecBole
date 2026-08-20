"""Pipeline for processing LastFM-1K dataset into RecBole format.

Mirrors the exact preprocessing steps, parameters, time-window filtering,
and sequence augmentation used for 30Music in process.py.
"""

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

# Time window filtering: 2007 and 2008 timestamp boundaries (UTC)
START_TIMESTAMP_2007 = 1167609600  # 2007-01-01 00:00:00 UTC
END_TIMESTAMP_2007 = 1199145599    # 2007-12-31 23:59:59 UTC (ostatni dzień 2007)

START_TIMESTAMP = 1199145600  # 2008-01-01 00:00:00 UTC
END_TIMESTAMP = 1230767999    # 2008-12-31 23:59:59 UTC

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
    """Parse CLI arguments and override corresponding module-level constants."""
    parser = argparse.ArgumentParser(
        description="Process LastFM-1K sessions into RecBole format."
    )
    for arg_name in _ARG_TO_GLOBAL:
        parser.add_argument(f"--{arg_name}", type=int)
    parser.add_argument("--all_splits", action="store_true", help="Process all 5 equal non-overlapping splits covering full dataset")
    parser.add_argument("--reinit", action="store_true", help="Force re-initializing raw sub-sessions from 2007-01 to 2008-12")

    args = parser.parse_args()

    g = globals()
    for arg_name, global_name in _ARG_TO_GLOBAL.items():
        value = getattr(args, arg_name)
        if value is not None:
            g[global_name] = value

    return args


def get_data_file_path(data_path, data_file, file_extension=".tsv"):
    """Return the full path for a data file with the given extension."""
    return os.path.join(data_path, f"{data_file}{file_extension}")


def get_line_count(file_path):
    """Return the number of lines in *file_path* safely across platforms."""
    try:
        return int(subprocess.check_output(["wc", "-l", file_path]).split()[0])
    except Exception:
        with open(file_path, "rb") as f:
            return sum(1 for _ in f)


def safe_copy(src, dst):
    """Copy *src* to *dst*, removing a read-only *dst* first if necessary."""
    if os.path.exists(dst):
        try:
            os.chmod(dst, stat.S_IWRITE)
            os.remove(dst)
        except OSError:
            pass
    shutil.copyfile(src, dst)


def _copy_processed_to_temp():
    """Copy the processed sessions file into the temp directory."""
    safe_copy(
        get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
        get_data_file_path(DATA_PATH_TEMP, DATA_FILE),
    )


def _remove_temp_file():
    """Remove the temporary sessions file if it exists."""
    path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    if os.path.exists(path):
        os.remove(path)


def _iter_session_lines(path, desc="Processing"):
    """Yield ``(parts, tracks)`` for each line in a sessions TSV file."""
    total = get_line_count(path)
    with open(path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=total, desc=desc):
            parts = line.strip().split("\t")
            tracks = json.loads(parts[3])
            yield parts, tracks


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
    if not text or text.isspace() or '\ufffd' in text:
        return True
    if sum(1 for c in text if c.isalpha()) < 2:
        return True
    if _UNKNOWN_PATTERNS.search(text) or _URL_PATTERN.search(text):
        return True
    return False


def _parse_iso_timestamp(ts_str):
    """Parse ISO 8601 timestamp string into POSIX epoch integer timestamp."""
    try:
        dt = datetime(
            int(ts_str[:4]), int(ts_str[5:7]), int(ts_str[8:10]),
            int(ts_str[11:13]), int(ts_str[14:16]), int(ts_str[17:19]),
            tzinfo=timezone.utc
        )
        return int(dt.timestamp())
    except Exception:
        return None


def initialize():
    """Parse raw LastFM-1K TSV file, group scrobbles by user, sort chronologically,
    split into sub-sessions on inactivity gap, and write initial processed TSV."""
    print("\nInitializing LastFM-1K data...")

    input_path = LASTFM_RAW_FILE
    raw_sessions_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw LastFM-1K dataset not found at {input_path}")

    # 1. Read all scrobbles and group by user
    total_lines = get_line_count(input_path)
    user_scrobbles = {}

    print(f"Reading scrobbles from {input_path}...")
    with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
        for line in tqdm(fin, total=total_lines, desc="Reading scrobbles"):
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue

            user_id = parts[0].strip()
            ts_str = parts[1].strip()
            artist_name = parts[3].strip()
            track_name = parts[5].strip()

            if not user_id or _is_noisy(artist_name) or _is_noisy(track_name):
                continue

            ts = _parse_iso_timestamp(ts_str)
            if ts is None:
                continue
            if not (START_TIMESTAMP <= ts <= END_TIMESTAMP):
                continue

            track_key = f"{artist_name}/_/{track_name}"

            if user_id not in user_scrobbles:
                user_scrobbles[user_id] = []
            user_scrobbles[user_id].append((ts, track_key))

    print(f"Loaded scrobbles for {len(user_scrobbles):,} users.")

    # 2. Build track whitelist map (unique track keys -> numeric IDs)
    print("Collecting unique tracks...")
    track_key_to_id = {}
    for scrobbles in user_scrobbles.values():
        for _, track_key in scrobbles:
            if track_key not in track_key_to_id:
                track_key_to_id[track_key] = len(track_key_to_id) + 1

    print(f"Found {len(track_key_to_id):,} unique tracks.")

    # Save initial raw track key mapping
    os.makedirs(DATA_PATH_RAW, exist_ok=True)
    with open(LASTFM_RAW_TRACKS_FILE, "w", encoding="utf-8") as fout:
        for track_key, track_id in sorted(track_key_to_id.items(), key=lambda x: x[1]):
            fout.write(f"{track_id}\t{track_key}\n")

    # 3. Process sessions per user
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    session_counter = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for user_id, scrobbles in tqdm(user_scrobbles.items(), desc="Generating sessions"):
            # Sort chronologically
            scrobbles.sort(key=lambda x: x[0])

            # De-duplicate near-simultaneous plays of the same track (< 10s gap)
            dedup_tracks = []
            last_seen_ps = {}
            for ts, track_key in scrobbles:
                tid = track_key_to_id[track_key]
                if tid in last_seen_ps and abs(ts - last_seen_ps[tid]) < 10:
                    continue
                last_seen_ps[tid] = ts
                dedup_tracks.append({"id": tid, "ps": ts})

            if not dedup_tracks:
                continue

            # Split into sub-sessions on inactivity gaps
            sub_sessions = _split_into_sub_sessions(dedup_tracks)

            for sub_idx, sub_session in enumerate(sub_sessions):
                if not (MIN_SESSION_LENGTH <= len(sub_session) <= MAX_SESSION_LENGTH):
                    continue

                playtime = sub_session[-1]["ps"] - sub_session[0]["ps"]
                if not (MIN_SESSION_PLAYTIME <= playtime <= MAX_SESSION_PLAYTIME):
                    continue

                session_counter += 1
                session_timestamp = sub_session[0]["ps"]
                new_session_id = f"{session_counter}_{sub_idx}"

                # Store track playstart relative to session timestamp
                rel_sub_session = [
                    {"id": t["id"], "ps": t["ps"] - session_timestamp}
                    for t in sub_session
                ]

                fout.write(
                    f"{new_session_id}\t{session_timestamp}\t{user_id}"
                    f"\t{json.dumps(rel_sub_session, separators=(',', ':'))}\n"
                )

    print(f"Created {session_counter:,} valid sub-sessions.")

    safe_copy(
        get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE),
        raw_sessions_path,
    )


def _split_into_sub_sessions(tracks):
    """Split a chronologically sorted track list into sub-sessions.

    A new sub-session starts when the gap between consecutive play-starts
    exceeds ``_SESSION_INACTIVITY_GAP`` seconds.
    """
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


def get_min_max_timestamps():
    """Find minimum and maximum session timestamps in raw sessions file."""
    input_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    min_ts = float("inf")
    max_ts = 0
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.split("\t", 2)
            if len(parts) >= 2:
                try:
                    ts = int(parts[1])
                    if ts < min_ts:
                        min_ts = ts
                    if ts > max_ts:
                        max_ts = ts
                except ValueError:
                    pass
    return int(min_ts), int(max_ts)


def get_max_timestamp():
    """Find maximum session timestamp in raw sessions file."""
    _, max_ts = get_min_max_timestamps()
    return max_ts


def get_5_equal_splits():
    """Divide full dataset time range [min_ts, max_ts] into 5 equal non-overlapping intervals,
    returning (days_from_max, days_to_max) tuples relative to max_ts."""
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
    """Keep only sessions whose timestamp falls within
    ``[MAX_TIMESTAMP - days_from_max*86400, MAX_TIMESTAMP - days_to_max*86400]``.
    """
    if days_from_max is None:
        days_from_max = DAYS_FROM_MAX
    if days_to_max is None:
        days_to_max = DAYS_TO_MAX

    if max_timestamp is None:
        max_timestamp = get_max_timestamp()

    print(f"\nFiltering sessions: last {days_from_max} to {days_to_max} days from max timestamp ({max_timestamp})...")

    input_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)

    lower_bound = max_timestamp - days_from_max * 86400
    upper_bound = max_timestamp - days_to_max * 86400

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
    """Remove tracks not present in the ``lastfm_tracks.tsv`` whitelist, then
    discard sessions that fall outside the allowed length range."""
    print("\nFiltering tracks by lastfm_tracks.tsv whitelist...")

    input_path = get_data_file_path(DATA_PATH_TEMP, DATA_FILE)
    output_path = get_data_file_path(DATA_PATH_PROCESSED, DATA_FILE)
    tracks_file = LASTFM_TRACKS_FILE

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
    tracks_source = LASTFM_TRACKS_FILE
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


def make_track_names_file():
    """Build ``lastfm_tracks.tsv`` in the raw directory, keeping only tracks
    with at least ``MIN_TRACK_PLAYCOUNT`` plays and non-noisy names."""
    print("\nCreating track names file...")

    sessions_path = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    tracks_source = LASTFM_RAW_TRACKS_FILE
    output_path = LASTFM_TRACKS_FILE

    # Count track occurrences across all sessions
    track_counts = Counter()
    for _parts, session_tracks in _iter_session_lines(sessions_path, desc="Counting tracks in sessions"):
        for t in session_tracks:
            track_counts[int(t["id"])] += 1

    whitelisted_ids = {tid for tid, count in track_counts.items() if count >= MIN_TRACK_PLAYCOUNT}
    print(f"Found {len(track_counts):,} unique tracks, {len(whitelisted_ids):,} with >= {MIN_TRACK_PLAYCOUNT} plays")

    tracks = {}
    with open(tracks_source, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                continue
            tid = int(parts[0])
            if tid in whitelisted_ids:
                tracks[tid] = parts[1]

    with open(output_path, "w", encoding="utf-8") as fout:
        for tid, name in sorted(tracks.items()):
            fout.write(f"{tid}\t{name}\n")

    print(f"Saved {len(tracks):,} whitelisted tracks to {output_path}")


def _parse_tags_json(data, min_weight=_MIN_TAG_WEIGHT):
    """Extract a flat list of hyphenated tag strings from a JSON-parsed
    tag structure, filtering by weight >= min_weight."""
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
    """Return *tags* with duplicates removed, preserving insertion order."""
    return list(dict.fromkeys(t for t in tags if t))


def _normalize_artist_name(name):
    """Normalize an artist name into a safe RecBole token."""
    name = name.strip().lower()
    name = re.sub(r'[\s/\\,;:&+\'"!()\[\]{}]+', '-', name)
    name = re.sub(r'-{2,}', '-', name)
    return name.strip('-') or "unknown-artist"


def _load_artist_tags(artist_tags_path):
    """Load the artist-tags mapping from ``artists_tags.tsv``."""
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
                    # Also index by normalized artist name if column 1 is artist name
                    if len(parts) >= 2:
                        norm_artist = _normalize_artist_name(parts[1])
                        if norm_artist:
                            artist_tags[norm_artist] = " ".join(tags)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                pass

    return artist_tags


def make_item_file(alias):
    """Create the ``.item`` file mapping track IDs to artist tags and track tags."""
    print("\nCreating .item file...")

    artist_tags_path = os.path.join("dataset", "artists_tags_all.tsv")
    tracks_path = os.path.join("dataset", alias, "tracks.tsv")
    output_path = os.path.join("dataset", alias, f"{alias}.item")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    artist_tags = _load_artist_tags(artist_tags_path)

    try:
        total_tracks = get_line_count(tracks_path)
    except Exception:
        total_tracks = None

    n_with_artist_tags = 0
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
                      or artist_tags.get(artist)
                      or "unknown")

            if a_tags != "unknown":
                n_with_artist_tags += 1
            else:
                n_fallback += 1

            # Fallback track_tags to artist_tags
            t_tags = a_tags

            fout.write(f"{track_id}\t{a_tags}\t{t_tags}\n")

    print(f"Item tags created: {n_with_artist_tags:,} tracks with artist tags, "
          f"{n_fallback:,} fell back to 'unknown'.")


def get_dataset_name(prefix="lastfm1k__"):
    """Build a descriptive dataset name encoding the current filter settings."""
    name_parts = [
        f"days[{DAYS_FROM_MAX}-{DAYS_TO_MAX}]",
        f"pcount[{MIN_TRACK_PLAYCOUNT}]",
        f"ptime[{MIN_SESSION_PLAYTIME}-{MAX_SESSION_PLAYTIME}]",
        f"length[{MIN_SESSION_LENGTH}-{MAX_SESSION_LENGTH}]",
        f"recent[{MAX_SESSION_RECENT_TRACKS}]",
    ]
    return prefix + "_".join(name_parts)


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
    """Create sequential augmentation (``item_id_list``) from raw interactions."""
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
    """Write an augmented DataFrame to ``.inter`` format."""
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

    df = pd.read_csv(inter_path, sep="\t", header=0, names=col_names, dtype=str)

    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(float)

    print(f"  Total interactions: {len(df):,}")
    print(f"  Total sessions:     {df['session_id'].nunique():,}")

    # -- Temporal split -----------------------------------------------------
    print("\n--- Splitting sessions temporally ---")
    split_sets = split_sessions_temporal(df, "session_id", "timestamp", ratios)
    split_names = ("train", "valid", "test")

    for name, sess in zip(split_names, split_sets):
        print(f"  {name.capitalize()} sessions: {len(sess):,}")

    split_dfs = [df[df["session_id"].isin(s)] for s in split_sets]

    # -- Augmentation & writing ---------------------------------------------
    print(f"\n--- Augmenting sequences (max_seq_len={max_seq_len}) ---")
    output_dir = os.path.join("dataset", alias)

    for name, split_df in zip(split_names, split_dfs):
        aug = augment_sessions(split_df, "session_id", "item_id", "timestamp", max_seq_len)
        print(f"  {name.capitalize()} augmented rows: {len(aug):,}")
        output_path = os.path.join(output_dir, f"{alias}.{name}.inter")
        write_benchmark_inter(aug, output_path, "session_id", "item_id", "timestamp")
        print(f"  Written: {os.path.basename(output_path)}")


def process_single_split(days_from_max, days_to_max, max_ts):
    global DAYS_FROM_MAX, DAYS_TO_MAX
    DAYS_FROM_MAX = days_from_max
    DAYS_TO_MAX = days_to_max
    dataset_name = get_dataset_name("lastfm1k__")
    print(f"\n========================================================")
    print(f"Processing split: {dataset_name}")
    print(f"========================================================")

    filter_by_time_window(days_from_max=days_from_max, days_to_max=days_to_max, max_timestamp=max_ts)
    _copy_processed_to_temp()
    filter_tracks_by_playcount()
    _copy_processed_to_temp()
    make_inter_file(dataset_name)
    make_tracks_file(dataset_name)
    make_item_file(dataset_name)
    make_benchmark_splits(dataset_name)
    _remove_temp_file()


def main():
    """Run the full processing pipeline for LastFM-1K across full dataset."""
    args = parse_args()

    raw_sessions = get_data_file_path(DATA_PATH_RAW, DATA_FILE)
    if getattr(args, "reinit", False) or not os.path.exists(raw_sessions) or not os.path.exists(LASTFM_RAW_TRACKS_FILE):
        initialize()

    if getattr(args, "reinit", False) or not os.path.exists(LASTFM_TRACKS_FILE):
        make_track_names_file()

    min_ts, max_ts = get_min_max_timestamps()

    if getattr(args, "all_splits", False):
        splits = get_5_equal_splits()
        print(f"\nProcessing all 5 equal non-overlapping splits covering full dataset ({min_ts} to {max_ts})...")
        for d_from, d_to in splits:
            process_single_split(d_from, d_to, max_ts)
    else:
        process_single_split(DAYS_FROM_MAX, DAYS_TO_MAX, max_ts)


if __name__ == "__main__":
    main()