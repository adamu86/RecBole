#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt analizujący ile unikalnych utworów (track_id) odpada z surowego zbioru 30Music 
(sessions.idomaar) po:
1. Zastosowaniu okna czasowego (np. ostatnie 365 do 65 dni od MAX_TIMESTAMP)
2. Usunięciu sesji o długości 1 (MIN_SESSION_LENGTH >= 2)
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from collections import Counter
from tqdm import tqdm

try:
    import orjson as fast_json
except ImportError:
    fast_json = json


def format_ts(ts):
    if ts is None:
        return "N/A"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def analyze_track_drop(file_path, days_from_max=365, days_to_max=65, max_timestamp=1421745720):
    lower_bound = max_timestamp - days_from_max * 86400
    upper_bound = max_timestamp - days_to_max * 86400

    print("=" * 80)
    print(" ANALIZA UBYTKU UTWORÓW (TRACK LOSS ANALYSIS)")
    print("=" * 80)
    print(f"Plik wejściowy:     {file_path}")
    print(f"Okno czasowe:       [{format_ts(lower_bound)} .. {format_ts(upper_bound)}]")
    print(f"Minimalna długość:  >= 2 (usuwanie sesji o długości 1)")
    print("=" * 80 + "\n")

    _PREFIX = "event.session\t"
    _PREFIX_LEN = len(_PREFIX)

    raw_tracks_set = set()
    raw_sessions_cnt = 0
    raw_interactions_cnt = 0

    window_tracks_set = set()
    window_sessions_cnt = 0
    window_interactions_cnt = 0

    len1_sessions_cnt = 0
    len1_tracks_set = set()

    filtered_tracks_set = set()
    filtered_sessions_cnt = 0
    filtered_interactions_cnt = 0

    print(f"Parsowanie pliku: {file_path}...")
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Parsowanie sessions.idomaar"):
            if not line.startswith(_PREFIX):
                continue
            line_body = line[_PREFIX_LEN:]

            # Ekstrakcja timestampu
            idx1 = line_body.find("\t")
            if idx1 == -1:
                continue
            idx2 = line_body.find("\t", idx1 + 1)
            if idx2 == -1:
                continue

            try:
                ts = int(line_body[idx1 + 1 : idx2])
            except ValueError:
                ts = None

            split_idx = line_body.find("} {")
            if split_idx == -1:
                continue

            objects_str = line_body[split_idx + 2:].strip()

            try:
                session_objects = fast_json.loads(objects_str)
                tracks = session_objects.get("objects", [])
                track_ids = [t["id"] for t in tracks if "id" in t]
            except Exception:
                continue

            if not track_ids:
                continue

            n_tracks = len(track_ids)
            raw_sessions_cnt += 1
            raw_interactions_cnt += n_tracks
            raw_tracks_set.update(track_ids)

            # Sprawdzenie okna czasowego
            if ts is not None and lower_bound <= ts <= upper_bound:
                window_sessions_cnt += 1
                window_interactions_cnt += n_tracks
                window_tracks_set.update(track_ids)

                if n_tracks == 1:
                    len1_sessions_cnt += 1
                    len1_tracks_set.update(track_ids)
                else:
                    filtered_sessions_cnt += 1
                    filtered_interactions_cnt += n_tracks
                    filtered_tracks_set.update(track_ids)

    # Obliczenia ubytków
    raw_unique_tracks = len(raw_tracks_set)
    window_unique_tracks = len(window_tracks_set)
    filtered_unique_tracks = len(filtered_tracks_set)

    dropped_by_window = raw_tracks_set - window_tracks_set
    dropped_by_len1_in_window = window_tracks_set - filtered_tracks_set
    total_dropped = raw_tracks_set - filtered_tracks_set

    tracks_only_in_len1 = len1_tracks_set - filtered_tracks_set

    print("\n" + "=" * 80)
    print(" PODSUMOWANIE ANALIZY UBYTKU UTWORÓW I SESJI")
    print("=" * 80)
    
    print("\n1. LICZBA UNIKALNYCH UTWORÓW (TRACK IDs):")
    print(f"  • W pełnym surowym zbiorze (0 filtrów):    {raw_unique_tracks:>12,}")
    print(f"  • W oknie czasowym (wszystkie sesje):      {window_unique_tracks:>12,}  (Spadek o {len(dropped_by_window):,} / {len(dropped_by_window)/raw_unique_tracks*100:.2f}%)")
    print(f"  • W oknie czasowym (sesje length >= 2):    {filtered_unique_tracks:>12,}  (Spadek o {len(dropped_by_len1_in_window):,} / {len(dropped_by_len1_in_window)/window_unique_tracks*100:.2f}% względem okna)")
    print(f"  -------------------------------------------------------------------------")
    print(f"  • ŁĄCZNY UBYTEK UTWORÓW (Window + Len>=2): {len(total_dropped)::>12,}  ({len(total_dropped)/raw_unique_tracks*100:.2f}% surowego zbioru)")

    print("\n2. SZCZEGÓŁY DLA SESJI O DŁUGOŚCI 1 (W OKNIE CZASOWYM):")
    print(f"  • Liczba sesji o długości == 1 w oknie:    {len1_sessions_cnt:>12,}  ({len1_sessions_cnt/window_sessions_cnt*100:.2f}% sesji w oknie)")
    print(f"  • Utwory występujące w sesjach length == 1: {len(len1_tracks_set):>12,}")
    print(f"  • Utwory występujące WYŁĄCZNIE w len == 1: {len(tracks_only_in_len1):>12,}  (te utwory przepadają po usunięciu len=1)")

    print("\n3. PODSUMOWANIE SESJI I INTERAKCJI:")
    print(f"  • Surowe sesje (pełne):                  {raw_sessions_cnt:>12,}")
    print(f"  • Sesje w oknie czasowym (wszystkie):     {window_sessions_cnt:>12,}  ({window_sessions_cnt/raw_sessions_cnt*100:.2f}% surowych)")
    print(f"  • Sesje w oknie czasowym (length >= 2):   {filtered_sessions_cnt:>12,}  ({filtered_sessions_cnt/window_sessions_cnt*100:.2f}% sesji w oknie)")
    print(f"  • Odzucone sesje length == 1 w oknie:     {len1_sessions_cnt:>12,}  ({len1_sessions_cnt/window_sessions_cnt*100:.2f}% sesji w oknie)")
    print("-" * 80)
    print(f"  • Surowe odsłuchania (pełne):             {raw_interactions_cnt:>12,}")
    print(f"  • Odsłuchania w oknie czasowym:           {window_interactions_cnt:>12,}")
    print(f"  • Odsłuchania w oknie czasowym (len>=2):  {filtered_interactions_cnt:>12,}")
    print(f"  • Utracone odsłuchania (len==1 w oknie):   {len1_sessions_cnt:>12,}  ({len1_sessions_cnt/window_interactions_cnt*100:.2f}% odsłuchań w oknie)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analiza ubytku utworów po oknie czasowym oraz filtrowaniu sesji length >= 2."
    )
    parser.add_argument(
        "--file", "-f", type=str, default="dataset_raw/sessions.idomaar",
        help="Ścieżka do surowego pliku sessions.idomaar."
    )
    parser.add_argument(
        "--days-from-max", type=int, default=365,
        help="Liczba dni od max timestamp dla dolnej granicy okna czasowego (domyślnie: 365)."
    )
    parser.add_argument(
        "--days-to-max", type=int, default=65,
        help="Liczba dni od max timestamp dla górnej granicy okna czasowego (domyślnie: 65)."
    )
    parser.add_argument(
        "--max-timestamp", type=int, default=1421745720,
        help="Maksymalny timestamp w zbiorze (domyślnie: 1421745720)."
    )

    args = parser.parse_args()
    file_path = args.file

    if not os.path.exists(file_path):
        alt_path = os.path.join("dataset_raw", os.path.basename(file_path))
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            print(f"[BŁĄD] Plik nie istnieje: {file_path}")
            sys.exit(1)

    analyze_track_drop(
        file_path,
        days_from_max=args.days_from_max,
        days_to_max=args.days_to_max,
        max_timestamp=args.max_timestamp,
    )


if __name__ == "__main__":
    main()
