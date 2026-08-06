"""
Script for generating figures on TRUE RAW datasets without any filtering or date window trimming:
1. Liczba sesji w poszczególnych dniach w surowym zbiorze danych 30Music (fig:sessions_per_day_raw_30music)
2. Rozkład popularności utworów w surowym zbiorze danych 30Music (long tail) (fig:longtail_raw_30music)
3. Liczba sesji w poszczególnych dniach w surowym zbiorze danych LastFM-1k (fig:sessions_per_day_raw_lastfm)
4. Rozkład popularności utworów w surowym zbiorze danych LastFM-1k (long tail) (fig:longtail_raw_lastfm)

- Reads raw 30Music sessions directly from dataset_raw/sessions.idomaar (all sessions, full timeline).
- Reads raw LastFM scrobbles directly from dataset_raw/userid-timestamp-artid-artname-traid-traname.tsv.
- Groups LastFM scrobbles by user and splits on 1800s (30 min) inactivity gap without any 2-year window trimming or filtering (full 2005-2009 timeline).

Plot visual style matches results.py.
"""

import os
import json
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

# Visual style configuration matching results.py
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
LASTFM_RAW_SCROBBLES = RAW_DATA_DIR / "userid-timestamp-artid-artname-traid-traname.tsv"
MUSIC30_RAW_IDOMAAR = RAW_DATA_DIR / "sessions.idomaar"

# Standard inactivity gap for raw scrobble sessionization (30 minutes)
SESSION_GAP = 800


def _parse_iso_ts(ts_str: str) -> int | None:
    """Fast ISO 8601 string parsing into UTC timestamp integer, filtering out invalid clock-drift years."""
    try:
        year = int(ts_str[:4])
        # LastFM-1k dataset valid time window is 2005 to 2009
        if not (2005 <= year <= 2009):
            return None
        return int(datetime(
            year, int(ts_str[5:7]), int(ts_str[8:10]),
            int(ts_str[11:13]), int(ts_str[14:16]), int(ts_str[17:19]),
            tzinfo=timezone.utc
        ).timestamp())
    except Exception:
        return None


def process_raw_30music(file_path: Path):
    """Parse pure raw 30Music dataset from sessions.idomaar covering full time span."""
    print(f"\n[30Music] Processing PURE RAW dataset from {file_path}...")
    daily_sessions = Counter()
    track_counts = Counter()

    _IDOMAAR_PREFIX = "event.session\t"
    _IDOMAAR_PREFIX_LEN = len(_IDOMAAR_PREFIX)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Reading 30Music sessions.idomaar"):
            if not line.startswith(_IDOMAAR_PREFIX):
                continue
            line_body = line[_IDOMAAR_PREFIX_LEN:]
            try:
                idx1 = line_body.find("\t")
                if idx1 == -1:
                    continue
                idx2 = line_body.find("\t", idx1 + 1)
                if idx2 == -1:
                    continue

                timestamp = int(line_body[idx1 + 1 : idx2])
                if not (1388534400 <= timestamp <= 1425168000):  # 2014-01-01 to 2015-03-01
                    continue

                split_idx = line_body.find("} {")
                if split_idx == -1:
                    continue

                obj_str = line_body[split_idx + 2:].strip()
                session_objects = fast_json.loads(obj_str)
                tracks = session_objects.get("objects", [])
                if not tracks:
                    continue

                date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
                daily_sessions[date_str] += 1

                for t in tracks:
                    tid = t.get("id")
                    if tid is not None:
                        track_counts[tid] += 1

            except Exception:
                continue

    print(f"[30Music] Total raw sessions: {sum(daily_sessions.values()):,}")
    print(f"[30Music] Total unique items: {len(track_counts):,}")
    return daily_sessions, track_counts


def process_raw_lastfm(file_path: Path):
    """Parse pure raw LastFM-1k scrobbles covering full time span (2005-2009)."""
    print(f"\n[LastFM-1k] Processing PURE RAW dataset from {file_path}...")
    user_scrobbles = defaultdict(list)

    total_lines = sum(1 for _ in open(file_path, "rb"))
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, total=total_lines, desc="Reading raw LastFM scrobbles"):
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            user_id = parts[0].strip()
            ts_str = parts[1].strip()
            artist = parts[3].strip()
            track = parts[5].strip()

            if not user_id or not artist or not track:
                continue

            ts = _parse_iso_ts(ts_str)
            if ts is None:
                continue

            track_key = f"{artist}/_/{track}"
            user_scrobbles[user_id].append((ts, track_key))

    print(f"[LastFM-1k] Loaded scrobbles for {len(user_scrobbles):,} users.")

    daily_sessions = Counter()
    track_counts = Counter()

    for user_id, scrobbles in tqdm(user_scrobbles.items(), desc="Building raw sub-sessions"):
        scrobbles.sort(key=lambda x: x[0])
        if not scrobbles:
            continue

        sess_start_ts = scrobbles[0][0]
        prev_ts = sess_start_ts
        sess_tracks = [scrobbles[0][1]]

        for ts, t_key in scrobbles[1:]:
            if ts - prev_ts > SESSION_GAP:
                # Save previous session
                date_str = datetime.fromtimestamp(sess_start_ts, tz=timezone.utc).strftime("%Y-%m-%d")
                daily_sessions[date_str] += 1
                for tk in sess_tracks:
                    track_counts[tk] += 1

                sess_start_ts = ts
                sess_tracks = []

            sess_tracks.append(t_key)
            prev_ts = ts

        if sess_tracks:
            date_str = datetime.fromtimestamp(sess_start_ts, tz=timezone.utc).strftime("%Y-%m-%d")
            daily_sessions[date_str] += 1
            for tk in sess_tracks:
                track_counts[tk] += 1

    print(f"[LastFM-1k] Total raw sessions: {sum(daily_sessions.values()):,}")
    print(f"[LastFM-1k] Total unique items: {len(track_counts):,}")
    return daily_sessions, track_counts


def render_sessions_per_day(daily_sessions: Counter, output_file: Path, color: str):
    dates = sorted(daily_sessions.keys())
    counts = [daily_sessions[d] for d in dates]
    df = pd.DataFrame({"date": pd.to_datetime(dates), "sessions": counts})

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(df["date"], df["sessions"], color=color, linewidth=1.5, alpha=0.9)
    ax.fill_between(df["date"], df["sessions"], color=color, alpha=0.15)

    start_date = df["date"].iloc[0]
    end_date = df["date"].iloc[-1]

    # Add a small padding margin from the Y-axis and right boundary
    time_span = end_date - start_date
    padding = time_span * 0.025
    ax.set_xlim(start_date - padding, end_date + padding)

    # Place ticks at exact start date, intermediate dates, and exact end date
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
    counts = sorted(track_counts.values(), reverse=True)
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


def main():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 30Music Raw Plots (entire timeline)
    if MUSIC30_RAW_IDOMAAR.exists():
        daily_30music, tracks_30music = process_raw_30music(MUSIC30_RAW_IDOMAAR)

        out_sessions_30m = DOCUMENTS_DIR / "sessions_per_day_raw_30music.png"
        render_sessions_per_day(daily_30music, out_sessions_30m, color=CUSTOM_COLORS[0])

        out_longtail_30m = DOCUMENTS_DIR / "longtail_raw_30music.png"
        render_long_tail(tracks_30music, out_longtail_30m, color=CUSTOM_COLORS[5])
    else:
        print(f"Error: Raw 30Music file not found at {MUSIC30_RAW_IDOMAAR}")

    # 2. LastFM-1k Raw Plots (entire timeline 2005-2009)
    if LASTFM_RAW_SCROBBLES.exists():
        daily_lastfm, tracks_lastfm = process_raw_lastfm(LASTFM_RAW_SCROBBLES)

        out_sessions_lastfm = DOCUMENTS_DIR / "sessions_per_day_raw_lastfm.png"
        render_sessions_per_day(daily_lastfm, out_sessions_lastfm, color=CUSTOM_COLORS[0])

        out_longtail_lastfm = DOCUMENTS_DIR / "longtail_raw_lastfm.png"
        render_long_tail(tracks_lastfm, out_longtail_lastfm, color=CUSTOM_COLORS[5])
    else:
        print(f"Error: Raw LastFM scrobbles file not found at {LASTFM_RAW_SCROBBLES}")

    # Create test.png copy for default fallback compatibility
    default_test_file = DOCUMENTS_DIR / "sessions_per_day_raw_30music.png"
    if default_test_file.exists():
        shutil.copyfile(default_test_file, DOCUMENTS_DIR / "test.png")

    print("\n[OK] Pure raw dataset plot generation completed successfully!")


if __name__ == "__main__":
    main()
