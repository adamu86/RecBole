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

SESSION_INACTIVITY_GAP = 800


def parse_iso_timestamp(timestamp_string: str) -> int | None:
    try:
        year = int(timestamp_string[:4])
        if not (2005 <= year <= 2009):
            return None
        return int(datetime(
            year,
            int(timestamp_string[5:7]),
            int(timestamp_string[8:10]),
            int(timestamp_string[11:13]),
            int(timestamp_string[14:16]),
            int(timestamp_string[17:19]),
            tzinfo=timezone.utc
        ).timestamp())
    except Exception:
        return None


def process_raw_30music(dataset_file_path: Path):
    print(f"\n[30Music] Processing dataset from {dataset_file_path}...")
    daily_session_counts = Counter()
    track_play_counts = Counter()
    session_length_counts = Counter()

    prefix = "event.session\t"
    prefix_length = len(prefix)

    with open(dataset_file_path, "r", encoding="utf-8", errors="replace") as file_handle:
        for line in tqdm(file_handle, desc="Reading 30Music sessions"):
            if not line.startswith(prefix):
                continue
            line_body = line[prefix_length:]
            try:
                first_tab_index = line_body.find("\t")
                if first_tab_index == -1:
                    continue
                second_tab_index = line_body.find("\t", first_tab_index + 1)
                if second_tab_index == -1:
                    continue

                timestamp = int(line_body[first_tab_index + 1 : second_tab_index])
                if not (1388534400 <= timestamp <= 1425168000):
                    continue

                json_start_index = line_body.find("} {")
                if json_start_index == -1:
                    continue

                json_object_string = line_body[json_start_index + 2:].strip()
                session_data = fast_json.loads(json_object_string)
                tracks = session_data.get("objects", [])
                if not tracks:
                    continue

                date_string = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
                daily_session_counts[date_string] += 1
                session_length_counts[len(tracks)] += 1

                for track in tracks:
                    track_id = track.get("id")
                    if track_id is not None:
                        track_play_counts[track_id] += 1

            except Exception:
                continue

    print(f"[30Music] Total sessions: {sum(daily_session_counts.values()):,}")
    print(f"[30Music] Total unique items: {len(track_play_counts):,}")
    return daily_session_counts, track_play_counts, session_length_counts


def process_raw_lastfm(dataset_file_path: Path):
    print(f"\n[LastFM-1k] Processing dataset from {dataset_file_path}...")
    user_scrobbles = defaultdict(list)

    total_lines = sum(1 for _ in open(dataset_file_path, "rb"))
    with open(dataset_file_path, "r", encoding="utf-8", errors="replace") as file_handle:
        for line in tqdm(file_handle, total=total_lines, desc="Reading LastFM scrobbles"):
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            user_id = parts[0].strip()
            timestamp_string = parts[1].strip()
            artist_name = parts[3].strip()
            track_name = parts[5].strip()

            if not user_id or not artist_name or not track_name:
                continue

            timestamp = parse_iso_timestamp(timestamp_string)
            if timestamp is None:
                continue

            track_key = f"{artist_name}/_/{track_name}"
            user_scrobbles[user_id].append((timestamp, track_key))

    print(f"[LastFM-1k] Loaded scrobbles for {len(user_scrobbles):,} users.")

    daily_session_counts = Counter()
    track_play_counts = Counter()
    session_length_counts = Counter()

    for user_id, scrobbles in tqdm(user_scrobbles.items(), desc="Building sub-sessions"):
        scrobbles.sort(key=lambda item: item[0])
        if not scrobbles:
            continue

        session_start_timestamp = scrobbles[0][0]
        previous_timestamp = session_start_timestamp
        current_session_tracks = [scrobbles[0][1]]

        for timestamp, track_key in scrobbles[1:]:
            if timestamp - previous_timestamp > SESSION_INACTIVITY_GAP:
                date_string = datetime.fromtimestamp(session_start_timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
                daily_session_counts[date_string] += 1
                session_length_counts[len(current_session_tracks)] += 1
                for item_key in current_session_tracks:
                    track_play_counts[item_key] += 1

                session_start_timestamp = timestamp
                current_session_tracks = []

            current_session_tracks.append(track_key)
            previous_timestamp = timestamp

        if current_session_tracks:
            date_string = datetime.fromtimestamp(session_start_timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
            daily_session_counts[date_string] += 1
            session_length_counts[len(current_session_tracks)] += 1
            for item_key in current_session_tracks:
                track_play_counts[item_key] += 1

    print(f"[LastFM-1k] Total sessions: {sum(daily_session_counts.values()):,}")
    print(f"[LastFM-1k] Total unique items: {len(track_play_counts):,}")
    return daily_session_counts, track_play_counts, session_length_counts


def render_sessions_per_day(daily_session_counts: Counter, output_file_path: Path, plot_color: str):
    dates_array = np.array(sorted(daily_session_counts.keys()))
    counts_array = np.array([daily_session_counts[date_key] for date_key in dates_array])
    dataframe = pd.DataFrame({"date": pd.to_datetime(dates_array), "sessions": counts_array})

    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    axes.plot(dataframe["date"], dataframe["sessions"], color=plot_color, linewidth=1.5, alpha=0.9)
    axes.fill_between(dataframe["date"], dataframe["sessions"], color=plot_color, alpha=0.15)

    start_date = dataframe["date"].iloc[0]
    end_date = dataframe["date"].iloc[-1]

    padding = (end_date - start_date) * 0.025
    axes.set_xlim(start_date - padding, end_date + padding)

    intermediate_ticks = pd.date_range(start_date, end_date, periods=6)[1:-1]
    all_ticks = [start_date] + list(intermediate_ticks) + [end_date]

    axes.set_xticks(all_ticks)
    axes.set_xticklabels([d.strftime("%Y-%m") for d in all_ticks], fontweight="bold", fontsize=10, rotation=45, ha="right")

    axes.set_xlabel("Data", fontweight="bold", fontsize=11, labelpad=6)
    axes.set_ylabel("Liczba sesji", fontweight="bold", fontsize=11, labelpad=6)
    axes.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, pos: f"{int(value):,}".replace(",", " ")))

    axes.grid(True, linestyle="--", alpha=0.4)
    axes.set_axisbelow(True)
    sns.despine(ax=axes, top=True, right=True)

    plt.tight_layout(pad=0.3)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"Generated plot: {output_file_path}")


def render_long_tail(track_play_counts: Counter, output_file_path: Path, plot_color: str):
    play_counts_array = np.fromiter(track_play_counts.values(), dtype=np.int64)
    sorted_counts = np.sort(play_counts_array)[::-1]
    ranks = np.arange(1, len(sorted_counts) + 1)

    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    axes.plot(ranks, sorted_counts, color=plot_color, linewidth=1.8)
    axes.fill_between(ranks, sorted_counts, color=plot_color, alpha=0.20)

    axes.set_yscale("log")

    axes.set_xlabel("Utwory (posortowane wg popularności)", fontweight="bold", fontsize=11, labelpad=6)
    axes.set_ylabel("Popularność (liczba odtworzeń)", fontweight="bold", fontsize=11, labelpad=6)

    axes.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, pos: f"{int(value):,}".replace(",", " ")))
    axes.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, pos: f"{int(value):g}".replace(".", ",")))

    axes.grid(True, linestyle="--", alpha=0.4, which="both")
    axes.set_axisbelow(True)
    sns.despine(ax=axes, top=True, right=True)

    plt.tight_layout(pad=0.3)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"Generated plot: {output_file_path}")


def render_session_lengths(session_length_counts: Counter, output_file_path: Path, plot_color: str, max_session_length: int = 100):
    length_indices = np.arange(1, max_session_length + 1)
    counts_array = np.zeros(max_session_length, dtype=np.int64)

    raw_lengths = np.fromiter(session_length_counts.keys(), dtype=np.int64)
    raw_counts = np.fromiter(session_length_counts.values(), dtype=np.int64)

    within_mask = (raw_lengths >= 1) & (raw_lengths < max_session_length)
    counts_array[raw_lengths[within_mask] - 1] = raw_counts[within_mask]

    over_max_count = raw_counts[raw_lengths >= max_session_length].sum()
    counts_array[-1] = over_max_count

    figure, axes = plt.subplots(figsize=(8.5, 4.5))
    axes.bar(length_indices, counts_array, color=plot_color, alpha=0.85, width=0.8, edgecolor="none")

    axes.set_xlabel("Długość sesji (liczba interakcji)", fontweight="bold", fontsize=11, labelpad=6)
    axes.set_ylabel("Liczba sesji", fontweight="bold", fontsize=11, labelpad=6)

    ticks = [1] + list(range(5, max_session_length + 1, 5))
    tick_labels = ["1"] + [str(tick) for tick in range(5, max_session_length, 5)] + [f"{max_session_length}+"]
    axes.set_xticks(ticks)
    axes.set_xticklabels(tick_labels, fontweight="bold", fontsize=9, rotation=45, ha="right")

    axes.set_xlim(0.2, max_session_length + 0.8)
    axes.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, pos: f"{int(value):,}".replace(",", " ")))

    axes.grid(True, linestyle="--", alpha=0.4, axis="y")
    axes.set_axisbelow(True)
    sns.despine(ax=axes, top=True, right=True)

    plt.tight_layout(pad=0.3)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"Generated plot: {output_file_path}")


def main():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if MUSIC30_RAW_IDOMAAR.exists():
        daily_30music, tracks_30music, lengths_30music = process_raw_30music(MUSIC30_RAW_IDOMAAR)

        output_sessions_per_day_30music = DOCUMENTS_DIR / "sessions_per_day_raw_30music.png"
        render_sessions_per_day(daily_30music, output_sessions_per_day_30music, plot_color=CUSTOM_COLORS[0])

        output_longtail_30music = DOCUMENTS_DIR / "longtail_raw_30music.png"
        render_long_tail(tracks_30music, output_longtail_30music, plot_color=CUSTOM_COLORS[5])

        output_session_lengths_30music = DOCUMENTS_DIR / "session_lengths_raw_30music.png"
        render_session_lengths(lengths_30music, output_session_lengths_30music, plot_color=CUSTOM_COLORS[1])
    else:
        print(f"Error: Raw 30Music file not found at {MUSIC30_RAW_IDOMAAR}")

    if LASTFM_RAW_SCROBBLES.exists():
        daily_lastfm, tracks_lastfm, lengths_lastfm = process_raw_lastfm(LASTFM_RAW_SCROBBLES)

        output_sessions_per_day_lastfm = DOCUMENTS_DIR / "sessions_per_day_raw_lastfm.png"
        render_sessions_per_day(daily_lastfm, output_sessions_per_day_lastfm, plot_color=CUSTOM_COLORS[0])

        output_longtail_lastfm = DOCUMENTS_DIR / "longtail_raw_lastfm.png"
        render_long_tail(tracks_lastfm, output_longtail_lastfm, plot_color=CUSTOM_COLORS[5])

        output_session_lengths_lastfm = DOCUMENTS_DIR / "session_lengths_raw_lastfm.png"
        render_session_lengths(lengths_lastfm, output_session_lengths_lastfm, plot_color=CUSTOM_COLORS[1])
    else:
        print(f"Error: Raw LastFM scrobbles file not found at {LASTFM_RAW_SCROBBLES}")

    default_test_file = DOCUMENTS_DIR / "sessions_per_day_raw_30music.png"
    if default_test_file.exists():
        shutil.copyfile(default_test_file, DOCUMENTS_DIR / "test.png")

    print("\n[OK] Pure raw dataset plot generation completed successfully!")


if __name__ == "__main__":
    main()
