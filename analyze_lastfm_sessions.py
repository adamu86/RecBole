"""Quick analysis of LastFM-1K session length distribution BEFORE filtering.

Re-parses raw scrobbles, splits into sub-sessions using the same 1800s gap,
and reports percentiles of session lengths without applying MIN/MAX filters.
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

RAW_FILE = "dataset_raw/userid-timestamp-artid-artname-traid-traname.tsv"
SESSION_GAP = 1800  # same as _SESSION_INACTIVITY_GAP


def _parse_ts(ts_str):
    try:
        return int(datetime(
            int(ts_str[:4]), int(ts_str[5:7]), int(ts_str[8:10]),
            int(ts_str[11:13]), int(ts_str[14:16]), int(ts_str[17:19]),
            tzinfo=timezone.utc
        ).timestamp())
    except Exception:
        return None


def main():
    if not os.path.exists(RAW_FILE):
        print(f"File not found: {RAW_FILE}")
        sys.exit(1)

    # 1. Read scrobbles grouped by user
    print("Reading scrobbles...")
    user_scrobbles = {}
    total = sum(1 for _ in open(RAW_FILE, "rb"))

    with open(RAW_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, total=total, desc="Parsing"):
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            uid = parts[0].strip()
            ts = _parse_ts(parts[1].strip())
            if ts is None:
                continue
            user_scrobbles.setdefault(uid, []).append(ts)

    print(f"Users: {len(user_scrobbles):,}")

    # 2. Split into sub-sessions and collect lengths
    all_lengths = []
    all_durations = []

    for uid, timestamps in tqdm(user_scrobbles.items(), desc="Building sessions"):
        timestamps.sort()
        # Split on gap
        if not timestamps:
            continue
        session_start = 0
        for i in range(1, len(timestamps)):
            if timestamps[i] - timestamps[i - 1] > SESSION_GAP:
                length = i - session_start
                duration = timestamps[i - 1] - timestamps[session_start]
                all_lengths.append(length)
                all_durations.append(duration)
                session_start = i
        # Last session
        length = len(timestamps) - session_start
        duration = timestamps[-1] - timestamps[session_start]
        all_lengths.append(length)
        all_durations.append(duration)

    lengths = np.array(all_lengths)
    durations = np.array(all_durations)

    # 3. Report
    print(f"\n{'='*60}")
    print(f"LastFM-1K Session Length Distribution (BEFORE filtering)")
    print(f"{'='*60}")
    print(f"Total sub-sessions: {len(lengths):,}")
    print(f"Total scrobbles:    {lengths.sum():,}")
    print()

    print("--- Session LENGTH (number of tracks) ---")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]:
        v = np.percentile(lengths, p)
        print(f"  P{p:>5}: {v:>8.0f}")
    print(f"  Mean:  {lengths.mean():>8.1f}")
    print(f"  Std:   {lengths.std():>8.1f}")
    print()

    print("--- Session DURATION (seconds) ---")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]:
        v = np.percentile(durations, p)
        print(f"  P{p:>5}: {v:>8.0f}s  ({v/60:>6.1f} min)")
    print(f"  Mean:  {durations.mean():>8.1f}s  ({durations.mean()/60:>6.1f} min)")
    print()

    # Sessions that would be kept vs filtered by current params
    kept_mask = (lengths >= 2) & (lengths <= 100) & (durations >= 30) & (durations <= 1_000_000)
    print("--- Filter impact (MIN_LEN=2, MAX_LEN=100, MIN_TIME=30s) ---")
    print(f"  Sessions kept:     {kept_mask.sum():>10,} ({kept_mask.mean()*100:.1f}%)")
    print(f"  Sessions removed:  {(~kept_mask).sum():>10,} ({(~kept_mask).mean()*100:.1f}%)")
    print()

    too_short = lengths < 2
    too_long = lengths > 100
    too_brief = durations < 30
    print(f"  Removed: length < 2:   {too_short.sum():>10,} ({too_short.mean()*100:.1f}%)")
    print(f"  Removed: length > 100: {too_long.sum():>10,} ({too_long.mean()*100:.1f}%)")
    print(f"  Removed: duration < 30s: {too_brief.sum():>10,} ({too_brief.mean()*100:.1f}%)")
    print()

    # What MAX_SESSION_LENGTH would cover 99th/99.5th percentile?
    for target_p in [95, 99, 99.5, 99.9]:
        val = np.percentile(lengths[lengths >= 2], target_p)
        print(f"  P{target_p} of sessions with len>=2: {val:.0f} tracks")


if __name__ == "__main__":
    main()
