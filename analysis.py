
import json
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

mpl.use("Agg")

COLOR = "#4a90e2"
sns.set_palette([COLOR])
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral", "serif"],
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = Path("documents")
RAW_DIR = Path("dataset_raw")
LASTFM_FILE = RAW_DIR / "userid-timestamp-artid-artname-traid-traname.tsv"
MUSIC30_FILE = RAW_DIR / "sessions.idomaar"

SESSION_INACTIVITY_GAP = 800

def load_30music(path: Path):
    daily = Counter()
    tracks = Counter()
    lengths = Counter()
    prefix = "event.session\t"
    plen = len(prefix)

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in tqdm(fh, desc="30Music sessions"):
            if not line.startswith(prefix):
                continue
            body = line[plen:]
            try:
                t1 = body.find("\t")
                t2 = body.find("\t", t1 + 1) if t1 != -1 else -1
                if t2 == -1:
                    continue

                ts = int(body[t1 + 1 : t2])

                jstart = body.find("} {")
                if jstart == -1:
                    continue

                session = fast_json.loads(body[jstart + 2 :].strip())
                objs = session.get("objects", [])
                if not objs:
                    continue

                day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                daily[day] += 1
                lengths[len(objs)] += 1
                for t in objs:
                    tid = t.get("id")
                    if tid is not None:
                        tracks[tid] += 1
            except Exception:
                continue

    return daily, tracks, lengths


def load_lastfm(path: Path):
    user_events: dict[str, list[tuple[int, str]]] = defaultdict(list)

    def parse_iso_timestamp(ts: str) -> int | None:
        try:
            year = int(ts[:4])
            if not (2005 <= year <= 2009):
                return None
            return int(datetime(
                year, int(ts[5:7]), int(ts[8:10]),
                int(ts[11:13]), int(ts[14:16]), int(ts[17:19]),
                tzinfo=timezone.utc,
            ).timestamp())
        except Exception:
            return None

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in tqdm(fh, desc="LastFM scrobbles"):
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            uid = parts[0].strip()
            ts_str = parts[1].strip()
            artist = parts[3].strip()
            track = parts[5].strip()
            if not uid or not artist or not track:
                continue
            ts = parse_iso_timestamp(ts_str)
            if ts is None:
                continue
            user_events[uid].append((ts, f"{artist}/_/{track}"))

    daily = Counter()
    tracks = Counter()
    lengths = Counter()

    for uid, evts in tqdm(user_events.items(), desc="Building sessions"):
        evts.sort()
        if not evts:
            continue
        sess_start = evts[0][0]
        prev_ts = sess_start
        sess_tracks: list[str] = [evts[0][1]]

        for ts, tk in evts[1:]:
            if ts - prev_ts > SESSION_INACTIVITY_GAP:
                day = datetime.fromtimestamp(sess_start, tz=timezone.utc).strftime("%Y-%m-%d")
                daily[day] += 1
                lengths[len(sess_tracks)] += 1
                for st in sess_tracks:
                    tracks[st] += 1
                sess_start = ts
                sess_tracks = []
            sess_tracks.append(tk)
            prev_ts = ts

        if sess_tracks:
            day = datetime.fromtimestamp(sess_start, tz=timezone.utc).strftime("%Y-%m-%d")
            daily[day] += 1
            lengths[len(sess_tracks)] += 1
            for st in sess_tracks:
                tracks[st] += 1

    return daily, tracks, lengths

def plot_longtail(track_counts: Counter, out: Path, color: str = COLOR):
    sorted_counts = np.sort(np.fromiter(track_counts.values(), dtype=np.int64))[::-1]
    ranks = np.arange(1, len(sorted_counts) + 1)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(ranks, sorted_counts, color=color, linewidth=1.8)
    ax.fill_between(ranks, sorted_counts, color=color, alpha=0.20)
    ax.set_yscale("log")

    ax.set_xlabel("Utwory (posortowane wg popularności)", fontweight="bold", fontsize=11, labelpad=6)
    ax.set_ylabel("Popularność (liczba odtworzeń)", fontweight="bold", fontsize=11, labelpad=6)

    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " ")))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):g}".replace(".", ",")))

    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout(pad=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def plot_sessions_per_day(daily_counts: Counter, out: Path, color: str = COLOR):
    dates = np.array(sorted(daily_counts.keys()))
    counts = np.array([daily_counts[d] for d in dates])
    df = pd.DataFrame({"date": pd.to_datetime(dates), "sessions": counts})

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(df["date"], df["sessions"], color=color, linewidth=1.5, alpha=0.9)
    ax.fill_between(df["date"], df["sessions"], color=color, alpha=0.15)

    start, end = df["date"].iloc[0], df["date"].iloc[-1]
    pad = (end - start) * 0.025
    ax.set_xlim(start - pad, end + pad)

    mid_ticks = pd.date_range(start, end, periods=6)[1:-1]
    all_ticks = [start] + list(mid_ticks) + [end]
    ax.set_xticks(all_ticks)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in all_ticks], fontweight="bold", fontsize=10, rotation=45, ha="right")

    ax.set_xlabel("Data", fontweight="bold", fontsize=11, labelpad=6)
    ax.set_ylabel("Liczba sesji", fontweight="bold", fontsize=11, labelpad=6)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " ")))

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout(pad=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def plot_session_lengths(length_counts: Counter, out: Path, color: str = COLOR, max_len: int = 100):
    length_indices = np.arange(1, max_len + 1)
    counts_array = np.zeros(max_len, dtype=np.int64)

    raw_lengths = np.fromiter(length_counts.keys(), dtype=np.int64)
    raw_counts = np.fromiter(length_counts.values(), dtype=np.int64)

    within_mask = (raw_lengths >= 1) & (raw_lengths < max_len)
    counts_array[raw_lengths[within_mask] - 1] = raw_counts[within_mask]
    counts_array[-1] = raw_counts[raw_lengths >= max_len].sum()

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(length_indices, counts_array, color=color, alpha=0.85, width=0.8, edgecolor="none")

    ax.set_xlabel("Długość sesji (liczba interakcji)", fontweight="bold", fontsize=11, labelpad=6)
    ax.set_ylabel("Liczba sesji", fontweight="bold", fontsize=11, labelpad=6)

    ticks = [1] + list(range(5, max_len + 1, 5))
    tick_labels = ["1"] + [str(t) for t in range(5, max_len, 5)] + [f"{max_len}+"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontweight="bold", fontsize=9, rotation=45, ha="right")
    ax.set_xlim(0.2, max_len + 0.8)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " ")))

    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout(pad=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def print_stats(name: str, daily: Counter, tracks: Counter, lengths: Counter):
    total_sessions = sum(daily.values())
    unique_items = len(tracks)
    total_interactions = sum(tracks.values())

    print(f"\n{name}")
    print(f"Interactions: {total_interactions:,}")
    print(f"    Sessions: {total_sessions:,}")
    print(f"      Tracks: {unique_items:,}")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_30, tracks_30, lengths_30 = load_30music(MUSIC30_FILE)
    print_stats("30Music", daily_30, tracks_30, lengths_30)
    plot_longtail(tracks_30, OUTPUT_DIR / "longtail_raw_30music.png")
    plot_sessions_per_day(daily_30, OUTPUT_DIR / "sessions_per_day_raw_30music.png")
    plot_session_lengths(lengths_30, OUTPUT_DIR / "session_lengths_raw_30music.png")

    print("\n\n\n")

    daily_lfm, tracks_lfm, lengths_lfm = load_lastfm(LASTFM_FILE)
    print_stats("LastFM-1k", daily_lfm, tracks_lfm, lengths_lfm)
    plot_longtail(tracks_lfm, OUTPUT_DIR / "longtail_raw_lastfm.png")
    plot_sessions_per_day(daily_lfm, OUTPUT_DIR / "sessions_per_day_raw_lastfm.png")
    plot_session_lengths(lengths_lfm, OUTPUT_DIR / "session_lengths_raw_lastfm.png")

