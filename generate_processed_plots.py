"""
Script for dynamically constructing PROCESSED datasets IN MEMORY from raw data files,
without saving any dataset files to disk:

Pipeline for 30Music Processed (In-Memory):
- Source: dataset_raw/sessions.idomaar
- 10s track deduplication & 800s sub-session splitting
- Base filters: MIN_LEN=2, MAX_LEN=100, MIN_PLAYTIME=30s, MAX_PLAYTIME=1_000_000s
- Whitelist filter: MIN_TRACK_PLAYCOUNT=25
- Time window filter: 300 days of 2014 (DAYS_FROM_MAX=365, DAYS_TO_MAX=65)

Pipeline for LastFM-1k Processed (In-Memory):
- Source: dataset_raw/userid-timestamp-artid-artname-traid-traname.tsv
- Time window: Strict 2 full years (2007-01-01 to 2008-12-31 UTC)
- Placeholder/noise filtering & 10s track deduplication
- 800s sub-session splitting
- Base filters: MIN_LEN=2, MAX_LEN=100, MIN_PLAYTIME=30s, MAX_PLAYTIME=1_000_000s
- Whitelist filter: MIN_TRACK_PLAYCOUNT=25

Computes statistics and renders 4 plots matching exact visual style.
"""

import os
import json
import re
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    import orjson as fast_json
except ImportError:
    fast_json = json

# Visual style configuration matching results.py & raw plots
CUSTOM_COLORS = ["#356070", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
sns.set_palette(CUSTOM_COLORS)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral", "serif"],
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

DOCUMENTS_DIR = Path("documents")
RAW_DATA_DIR = Path("dataset_raw")

RAW_30MUSIC_FILE = RAW_DATA_DIR / "sessions.idomaar"
RAW_LASTFM_FILE = RAW_DATA_DIR / "userid-timestamp-artid-artname-traid-traname.tsv"

TRACKS_30MUSIC_FILE = RAW_DATA_DIR / "tracks.tsv"
TRACKS_LASTFM_FILE = RAW_DATA_DIR / "lastfm_tracks.tsv"

# Common preprocessing parameters
MIN_TRACK_PLAYCOUNT = 25
MIN_SESSION_LENGTH = 2
MAX_SESSION_LENGTH = 100
MIN_SESSION_PLAYTIME = 30
MAX_SESSION_PLAYTIME = 1_000_000
SESSION_GAP = 800

# Noise filtering regex for LastFM
_UNKNOWN_PATTERNS = re.compile(
    r'(?i)^(\[?unknown\]?|\[?none\]?|\[?null\]?|\[?deleted\]?|\[?untagged\]?|n/a|na|none|null|\?+|<artista desconocido>|<nieznany wykonawca>)$'
)
_URL_PATTERN = re.compile(r'https?://|www\.')


def _is_noisy(text: str) -> bool:
    if not text or text.isspace() or '\ufffd' in text:
        return True
    if sum(1 for c in text if c.isalpha()) < 2:
        return True
    if _UNKNOWN_PATTERNS.search(text) or _URL_PATTERN.search(text):
        return True
    return False


def _parse_iso_ts(ts_str: str) -> int | None:
    try:
        year = int(ts_str[:4])
        if not (2005 <= year <= 2009):
            return None
        return int(datetime(
            year, int(ts_str[5:7]), int(ts_str[8:10]),
            int(ts_str[11:13]), int(ts_str[14:16]), int(ts_str[17:19]),
            tzinfo=timezone.utc
        ).timestamp())
    except Exception:
        return None


def split_into_sub_sessions(tracks, gap):
    if not tracks:
        return []
    sub_sessions = []
    current = [tracks[0]]
    for prev, cur in zip(tracks, tracks[1:]):
        if cur["ps"] - prev["ps"] > gap:
            sub_sessions.append(current)
            current = []
        current.append(cur)
    sub_sessions.append(current)
    return sub_sessions


def load_track_artist_map(tracks_file: Path):
    track_to_artist = {}
    if not tracks_file.exists():
        return track_to_artist
    with open(tracks_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                tid = parts[0].strip()
                artist = parts[1].split("/_/")[0].strip()
                track_to_artist[tid] = artist
                try:
                    track_to_artist[int(tid)] = artist
                except ValueError:
                    pass
    return track_to_artist


def build_processed_30music_in_memory():
    print(f"\n[30Music] Constructing PROCESSED dataset IN MEMORY from {RAW_30MUSIC_FILE}...")
    track_artist_map = load_track_artist_map(TRACKS_30MUSIC_FILE)

    parsed_sessions = []
    _IDOMAAR_PREFIX = "event.session\t"
    _IDOMAAR_PREFIX_LEN = len(_IDOMAAR_PREFIX)

    with open(RAW_30MUSIC_FILE, "r", encoding="utf-8", errors="replace") as fin:
        for line in tqdm(fin, desc="Reading 30Music raw idomaar"):
            if not line.startswith(_IDOMAAR_PREFIX):
                continue
            line_clean = line[_IDOMAAR_PREFIX_LEN:]
            try:
                parts = line_clean.split("\t", 2)
                if len(parts) < 2:
                    continue
                session_timestamp = int(parts[1])

                split_idx = line_clean.find("} {")
                if split_idx == -1:
                    continue
                session_objects = fast_json.loads(line_clean[split_idx + 2:].strip())

                user_id = session_objects["subjects"][0]["id"] if ("subjects" in session_objects and session_objects["subjects"]) else "unknown"

                raw_tracks = []
                last_seen_ps = {}
                for st in session_objects.get("objects", []):
                    tid = st.get("id")
                    ps = st.get("playstart")
                    if tid is None or ps is None:
                        continue
                    if tid in last_seen_ps and abs(ps - last_seen_ps[tid]) < 10:
                        continue
                    last_seen_ps[tid] = ps
                    raw_tracks.append({"id": tid, "ps": ps})

                if not raw_tracks:
                    continue

                raw_tracks.sort(key=lambda x: x["ps"])
                sub_sessions = split_into_sub_sessions(raw_tracks, SESSION_GAP)

                for sub in sub_sessions:
                    l = len(sub)
                    d = sub[-1]["ps"] - sub[0]["ps"]
                    if not (MIN_SESSION_LENGTH <= l <= MAX_SESSION_LENGTH):
                        continue
                    if not (MIN_SESSION_PLAYTIME <= d <= MAX_SESSION_PLAYTIME):
                        continue
                    sub_start_ts = session_timestamp + sub[0]["ps"]
                    parsed_sessions.append({
                        "user_id": user_id,
                        "length": l,
                        "duration": d,
                        "start_ts": sub_start_ts,
                        "tracks": sub,
                    })

            except Exception:
                continue

    # 1. Track playcount filter (MIN_TRACK_PLAYCOUNT=25)
    track_counts_base = Counter()
    for s in parsed_sessions:
        for t in s["tracks"]:
            track_counts_base[t["id"]] += 1

    whitelisted_tracks = set(tid for tid, c in track_counts_base.items() if c >= MIN_TRACK_PLAYCOUNT)

    kept_pc = []
    for s in parsed_sessions:
        filt = [t for t in s["tracks"] if t["id"] in whitelisted_tracks]
        if MIN_SESSION_LENGTH <= len(filt) <= MAX_SESSION_LENGTH:
            kept_pc.append({
                "user_id": s["user_id"],
                "start_ts": s["start_ts"] + (filt[0]["ps"] - s["tracks"][0]["ps"]),
                "tracks": filt,
            })

    # 2. Time window filter (300 days of 2014)
    MAX_TIMESTAMP = 1421745720
    DAYS_FROM_MAX = 365
    DAYS_TO_MAX = 65
    lower_bound = MAX_TIMESTAMP - DAYS_FROM_MAX * 86400
    upper_bound = MAX_TIMESTAMP - DAYS_TO_MAX * 86400

    final_sessions = [s for s in kept_pc if lower_bound <= s["start_ts"] <= upper_bound]

    # Collect statistics & counters
    daily_sessions = Counter()
    track_counts = Counter()
    artist_set = set()
    total_interactions = 0

    for s in final_sessions:
        date_str = datetime.fromtimestamp(s["start_ts"], tz=timezone.utc).strftime("%Y-%m-%d")
        daily_sessions[date_str] += 1
        total_interactions += len(s["tracks"])
        for t in s["tracks"]:
            tid = t["id"]
            track_counts[tid] += 1
            artist = track_artist_map.get(tid) or track_artist_map.get(str(tid))
            if artist:
                artist_set.add(artist)

    total_sessions = len(final_sessions)
    mean_length = total_interactions / total_sessions if total_sessions > 0 else 0.0

    stats = {
        "dataset_name": "30Music (Processed)",
        "total_sessions": total_sessions,
        "total_interactions": total_interactions,
        "mean_session_length": mean_length,
        "unique_tracks": len(track_counts),
        "unique_artists": len(artist_set),
    }

    return stats, daily_sessions, track_counts


def build_processed_lastfm_in_memory():
    print(f"\n[LastFM-1k] Constructing PROCESSED dataset IN MEMORY from {RAW_LASTFM_FILE}...")
    track_artist_map = load_track_artist_map(TRACKS_LASTFM_FILE)

    START_TIMESTAMP = 1167609600  # 2007-01-01 UTC
    END_TIMESTAMP = 1230767999    # 2008-12-31 UTC

    user_scrobbles = defaultdict(list)
    track_key_to_id = {}

    total_lines = sum(1 for _ in open(RAW_LASTFM_FILE, "rb"))
    with open(RAW_LASTFM_FILE, "r", encoding="utf-8", errors="replace") as fin:
        for line in tqdm(fin, total=total_lines, desc="Reading LastFM raw scrobbles"):
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            user_id = parts[0].strip()
            ts_str = parts[1].strip()
            artist = parts[3].strip()
            track = parts[5].strip()

            if not user_id or _is_noisy(artist) or _is_noisy(track):
                continue

            ts = _parse_iso_ts(ts_str)
            if ts is None or not (START_TIMESTAMP <= ts <= END_TIMESTAMP):
                continue

            track_key = f"{artist}/_/{track}"
            if track_key not in track_key_to_id:
                track_key_to_id[track_key] = len(track_key_to_id) + 1

            user_scrobbles[user_id].append((ts, track_key_to_id[track_key], artist))

    # Deduplicate near-simultaneous plays (< 10s) and split into sub-sessions
    parsed_sessions = []
    for user_id, scrobbles in user_scrobbles.items():
        scrobbles.sort(key=lambda x: x[0])
        dedup_tracks = []
        last_seen_ps = {}
        for ts, tid, art in scrobbles:
            if tid in last_seen_ps and abs(ts - last_seen_ps[tid]) < 10:
                continue
            last_seen_ps[tid] = ts
            dedup_tracks.append({"id": tid, "ps": ts, "artist": art})

        if not dedup_tracks:
            continue

        subs = split_into_sub_sessions(dedup_tracks, SESSION_GAP)
        for s in subs:
            l = len(s)
            d = s[-1]["ps"] - s[0]["ps"]
            if not (MIN_SESSION_LENGTH <= l <= MAX_SESSION_LENGTH):
                continue
            if not (MIN_SESSION_PLAYTIME <= d <= MAX_SESSION_PLAYTIME):
                continue
            parsed_sessions.append({
                "user_id": user_id,
                "length": l,
                "duration": d,
                "start_ts": s[0]["ps"],
                "tracks": s,
            })

    # Track playcount filter (MIN_TRACK_PLAYCOUNT=25)
    track_counts_base = Counter()
    for s in parsed_sessions:
        for t in s["tracks"]:
            track_counts_base[t["id"]] += 1

    whitelisted_tracks = set(tid for tid, c in track_counts_base.items() if c >= MIN_TRACK_PLAYCOUNT)

    final_sessions = []
    for s in parsed_sessions:
        filt = [t for t in s["tracks"] if t["id"] in whitelisted_tracks]
        if MIN_SESSION_LENGTH <= len(filt) <= MAX_SESSION_LENGTH:
            final_sessions.append({
                "user_id": s["user_id"],
                "start_ts": filt[0]["ps"],
                "tracks": filt,
            })

    # Collect statistics & counters
    daily_sessions = Counter()
    track_counts = Counter()
    artist_set = set()
    total_interactions = 0

    for s in final_sessions:
        date_str = datetime.fromtimestamp(s["start_ts"], tz=timezone.utc).strftime("%Y-%m-%d")
        daily_sessions[date_str] += 1
        total_interactions += len(s["tracks"])
        for t in s["tracks"]:
            tid = t["id"]
            track_counts[tid] += 1
            artist = t.get("artist") or track_artist_map.get(tid) or track_artist_map.get(str(tid))
            if artist:
                artist_set.add(artist)

    total_sessions = len(final_sessions)
    mean_length = total_interactions / total_sessions if total_sessions > 0 else 0.0

    stats = {
        "dataset_name": "LastFM-1k (Processed)",
        "total_sessions": total_sessions,
        "total_interactions": total_interactions,
        "mean_session_length": mean_length,
        "unique_tracks": len(track_counts),
        "unique_artists": len(artist_set),
    }

    return stats, daily_sessions, track_counts


def render_sessions_per_day(
    daily_sessions: Counter,
    output_file: Path,
    color: str,
    target_start_date: str = None,
    target_end_date: str = None,
):
    dates = sorted(daily_sessions.keys())
    counts = [daily_sessions[d] for d in dates]
    df = pd.DataFrame({"date": pd.to_datetime(dates), "sessions": counts})

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(df["date"], df["sessions"], color=color, linewidth=1.5, alpha=0.9)
    ax.fill_between(df["date"], df["sessions"], color=color, alpha=0.15)

    if target_start_date is not None:
        start_date = pd.to_datetime(target_start_date)
    else:
        start_date = df["date"].iloc[0]

    if target_end_date is not None:
        end_date = pd.to_datetime(target_end_date)
    else:
        end_date = df["date"].iloc[-1]

    # Add small padding margin from Y-axis and right edge
    time_span = end_date - start_date
    padding = time_span * 0.025
    ax.set_xlim(start_date - padding, end_date + padding)

    # Place ticks at start date, intermediate dates, and end date
    num_intermediate = 4
    intermediate_ticks = pd.date_range(start_date, end_date, periods=num_intermediate + 2)[1:-1]
    all_ticks = [start_date] + list(intermediate_ticks) + [end_date]

    ax.set_xticks(all_ticks)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in all_ticks], fontweight="bold", fontsize=10, rotation=45, ha="right")

    ax.set_xlabel("Data", fontweight="bold", fontsize=11, labelpad=6)
    ax.set_ylabel("Liczba sesji", fontweight="bold", fontsize=11, labelpad=6)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda val, pos: f"{int(val):,}".replace(",", " ")))

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout(pad=0.3)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Generated plot: {output_file}")


def render_long_tail(track_counts: Counter, output_file: Path, color: str):
    counts = sorted([c for c in track_counts.values() if c > 0], reverse=True)
    if not counts:
        print(f"Warning: No valid track counts > 0 for {output_file}. Skipping plot.")
        return
    ranks = np.arange(1, len(counts) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ranks, counts, color=color, linewidth=1.8)
    ax.fill_between(ranks, counts, color=color, alpha=0.20)

    ax.set_yscale("log")

    ax.set_xlabel("Utwory (posortowane wg popularności)", fontweight="bold", fontsize=11, labelpad=6)
    ax.set_ylabel("Popularność (liczba odtworzeń)", fontweight="bold", fontsize=11, labelpad=6)

    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda val, pos: f"{int(val):,}".replace(",", " ")))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda val, pos: f"{int(val):g}".replace(".", ",")))

    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout(pad=0.3)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Generated plot: {output_file}")


def print_stats_table(stats_list):
    sep = "=" * 85
    print("\n" + sep)
    print(f"{'STATYSTYKI PRZETWORZONYCH ZBIORÓW DANYCH (PROCESSED DATASETS)':^85}")
    print(sep)
    header = f"{'Zbiór danych':<20} | {'Liczba sesji':<14} | {'Interakcje':<14} | {'Śr. dł. sesji':<14} | {'Unik. utwory':<13} | {'Unik. artyści'}"
    print(header)
    print("-" * 85)

    for st in stats_list:
        row = (
            f"{st['dataset_name']:<20} | "
            f"{st['total_sessions']:<14,} | "
            f"{st['total_interactions']:<14,} | "
            f"{st['mean_session_length']:<14.2f} | "
            f"{st['unique_tracks']:<13,} | "
            f"{st['unique_artists']:,}"
        )
        print(row)

    print(sep + "\n")


def main():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_summary = []

    # 1. 30Music Processed (In-Memory)
    if RAW_30MUSIC_FILE.exists():
        stats_30m, daily_30m, tracks_30m = build_processed_30music_in_memory()
        stats_summary.append(stats_30m)

        out_sessions_30m = DOCUMENTS_DIR / "sessions_per_day_processed_30music.png"
        render_sessions_per_day(
            daily_30m,
            out_sessions_30m,
            color=CUSTOM_COLORS[0],
            target_start_date="2014-01-20",
            target_end_date="2014-11-16",
        )

        out_longtail_30m = DOCUMENTS_DIR / "longtail_processed_30music.png"
        render_long_tail(tracks_30m, out_longtail_30m, color=CUSTOM_COLORS[5])
    else:
        print(f"Error: Raw 30Music file not found at {RAW_30MUSIC_FILE}")

    # 2. LastFM-1k Processed (In-Memory)
    if RAW_LASTFM_FILE.exists():
        stats_lastfm, daily_lastfm, tracks_lastfm = build_processed_lastfm_in_memory()
        stats_summary.append(stats_lastfm)

        out_sessions_lastfm = DOCUMENTS_DIR / "sessions_per_day_processed_lastfm.png"
        render_sessions_per_day(
            daily_lastfm,
            out_sessions_lastfm,
            color=CUSTOM_COLORS[0],
            target_start_date="2007-01-01",
            target_end_date="2008-12-31",
        )

        out_longtail_lastfm = DOCUMENTS_DIR / "longtail_processed_lastfm.png"
        render_long_tail(tracks_lastfm, out_longtail_lastfm, color=CUSTOM_COLORS[5])
    else:
        print(f"Error: Raw LastFM file not found at {RAW_LASTFM_FILE}")

    # Print summary statistics table to stdout
    print_stats_table(stats_summary)

    print("[OK] Processed dataset in-memory construction & plot generation completed successfully!")


if __name__ == "__main__":
    main()
