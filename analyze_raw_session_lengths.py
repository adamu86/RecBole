#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt do szczegółowej analizy rozkładu długości sesji w surowym zbiorze 30Music (sessions.idomaar).

Oblicza:
1. Dokładną liczbę i odsetek sesji o długości 1, 2, 3, 4, 5, ..., 15 oraz w przedziałach.
2. Statystyki skumulowane (<= 1, <= 2, <= 3, <= 4, <= 5, <= 10 itd.).
3. Percentyle długości sesji (P1, P5, P10, P25, P50, P75, P90, P95, P98, P99, P99.5, P99.9, Max).
4. Statystyki opisowe (średnia, mediana, odchylenie std, min, max, łączna liczba sesji, suma interakcji).
5. Opcjonalne filtrowanie oknem czasowym (np. ostatnie 365 do 65 dni względem MAX_TIMESTAMP).
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from collections import Counter
import numpy as np
from tqdm import tqdm

try:
    import orjson as fast_json
except ImportError:
    fast_json = json


def format_ts(ts):
    """Format Unix timestamp as ISO-like UTC string."""
    if ts is None:
        return "N/A"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_idomaar_sessions(file_path, lower_bound=None, upper_bound=None):
    """
    Parsuje surowy plik .idomaar (30Music dataset_raw/sessions.idomaar).
    Zlicza ścieżki wewnątrz każdej sesji z opcjonalnym filtrowaniem oknem czasowym.

    Args:
        file_path (str): Ścieżka do pliku .idomaar.
        lower_bound (int, optional): Minimalny timestamp (sekundy Unix).
        upper_bound (int, optional): Maksymalny timestamp (sekundy Unix).

    Returns:
        tuple: (all_lengths, window_lengths, min_ts_found, max_ts_found)
    """
    all_lengths = []
    window_lengths = []
    min_ts_found = None
    max_ts_found = None

    _PREFIX = "event.session\t"
    _PREFIX_LEN = len(_PREFIX)

    has_window = (lower_bound is not None) or (upper_bound is not None)

    print(f"Odczytywanie surowego pliku idomaar: {file_path}...")
    if has_window:
        l_str = format_ts(lower_bound) if lower_bound is not None else "-inf"
        u_str = format_ts(upper_bound) if upper_bound is not None else "+inf"
        print(f"Filtrowanie oknem czasowym: [{l_str} .. {u_str}]")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Parsowanie sessions.idomaar"):
            if not line.startswith(_PREFIX):
                continue
            line_body = line[_PREFIX_LEN:]

            # Ekstrakcja timestampu sesji (line_body: <session_id>\t<timestamp>\t...)
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

            if ts is not None:
                if min_ts_found is None or ts < min_ts_found:
                    min_ts_found = ts
                if max_ts_found is None or ts > max_ts_found:
                    max_ts_found = ts

            split_idx = line_body.find("} {")
            if split_idx == -1:
                continue

            objects_str = line_body[split_idx + 2:].strip()

            # Szybkie zliczanie wystąpień 'playstart' w obiekcie JSON
            cnt = objects_str.count('"playstart"')
            if cnt == 0:
                # Fallback do pełnego parsowania JSON w razie braku pola playstart
                try:
                    session_objects = fast_json.loads(objects_str)
                    tracks = session_objects.get("objects", [])
                    if tracks:
                        cnt = len(tracks)
                except Exception:
                    continue

            if cnt > 0:
                all_lengths.append(cnt)

                # Sprawdzenie czy sesja mieści się w oknie czasowym
                in_window = True
                if ts is not None:
                    if lower_bound is not None and ts < lower_bound:
                        in_window = False
                    if upper_bound is not None and ts > upper_bound:
                        in_window = False
                elif has_window:
                    in_window = False

                if in_window:
                    window_lengths.append(cnt)

    return all_lengths, window_lengths, min_ts_found, max_ts_found


def analyze_lengths(lengths, name="Surowy zbiór 30Music (sessions.idomaar)"):
    """
    Wyświetla pełny raport rozkładu długości sesji i percentyli.
    """
    if not lengths:
        print(f"\nBrak danych do analizy dla: {name}")
        return

    arr = np.array(lengths, dtype=np.int64)
    total_sessions = len(arr)
    total_events = int(arr.sum())

    print("\n" + "=" * 80)
    print(f" ANALIZA DŁUGOŚCI SESJI: {name}")
    print("=" * 80)
    print(f"Łączna liczba sesji:    {total_sessions:>12,}")
    print(f"Łączna liczba zdarzeń: {total_events:>12,}")
    print(f"Średnia długość sesji: {arr.mean():>12.2f}")
    print(f"Mediana długości sesji: {np.median(arr):>12.0f}")
    print(f"Odchylenie std:         {arr.std():>12.2f}")
    print(f"Najkrótsza sesja (Min): {arr.min():>12}")
    print(f"Najdłuższa sesja (Max): {arr.max():>12}")
    print("-" * 80)

    # 1. Dokładny rozkład długości 1, 2, 3, ..., 15 i przedziały
    counter = Counter(arr)
    print("\n1. DOKŁADNY ROZKŁAD DŁUGOŚCI SESJI:")
    print(f" {'Długość sesji':<18} | {'Liczba sesji':>12} | {'Udział (%)':>10} | {'Wizualizacja'}")
    print("-" * 75)

    max_exact = 15
    for l in range(1, max_exact + 1):
        cnt = counter[l]
        pct = (cnt / total_sessions) * 100
        bar = "█" * int(pct / 2)
        print(f"  Długość == {l:<8} | {cnt:>12,} | {pct:>9.2f}% | {bar}")

    # Przedziały dla większych długości
    ranges = [
        (16, 20),
        (21, 30),
        (31, 50),
        (51, 100),
        (101, float("inf"))
    ]

    for start, end in ranges:
        if end == float("inf"):
            cnt = sum(c for l, c in counter.items() if l >= start)
            label = f"  Długość > {start-1:<8}"
        else:
            cnt = sum(c for l, c in counter.items() if start <= l <= end)
            label = f"  Długość {start}-{end:<7}"

        pct = (cnt / total_sessions) * 100
        bar = "█" * int(pct / 2)
        print(f"{label} | {cnt:>12,} | {pct:>9.2f}% | {bar}")

    print("-" * 80)

    # 2. Skumulowane statystyki (≤ X)
    print("\n2. SKUMULOWANE STATYSTYKI SESJI (UDZIAŁ KROK PO KROKU):")
    cum_thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100]
    print(f" {'Długość ≤ X':<18} | {'Liczba sesji (≤ X)':>18} | {'Skumulowany Udział (%)':>22}")
    print("-" * 65)

    for th in cum_thresholds:
        cnt_le = sum(c for l, c in counter.items() if l <= th)
        pct_le = (cnt_le / total_sessions) * 100
        print(f"  Długość ≤ {th:<8} | {cnt_le:>18,} | {pct_le:>21.2f}%")

    print("-" * 80)

    # 3. Percentyle
    print("\n3. PERCENTYLE DŁUGOŚCI SESJI:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9, 100]
    p_values = np.percentile(arr, percentiles)

    print(f" {'Percentyl':<15} | {'Wartość (Długość sesji)':>24}")
    print("-" * 45)
    for p, val in zip(percentiles, p_values):
        name_p = f"P{p}" if p != 50 else "P50 (Mediana)"
        if p == 25:
            name_p = "P25 (Q1)"
        elif p == 75:
            name_p = "P75 (Q3)"
        elif p == 100:
            name_p = "P100 (Max)"
        print(f"  {name_p:<15} | {val:>24.1f}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analiza rozkładu długości sesji w surowym zbiorze 30Music (sessions.idomaar) z uwzględnieniem okna czasowego."
    )
    parser.add_argument(
        "--file", "-f", type=str, default="dataset_raw/sessions.idomaar",
        help="Ścieżka do surowego pliku sessions.idomaar (domyślnie: dataset_raw/sessions.idomaar)."
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
        help="Maksymalny timestamp w zbiorze 30Music (domyślnie: 1421745720)."
    )
    parser.add_argument(
        "--min-timestamp", type=int, default=None,
        help="Bezpośredni minimalny timestamp w sekundach (nadpisuje obliczenia z --days-from-max)."
    )
    parser.add_argument(
        "--no-time-window", action="store_true",
        help="Wyłącz filtrowanie oknem czasowym i przeanalizuj wyłącznie pełny surowy zbiór."
    )
    parser.add_argument(
        "--filter-only", action="store_true",
        help="Wyświetl wyłącznie raport dla sesji z oknem czasowym (pomiń pełny surowy raport)."
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

    if args.no_time_window:
        lower_bound = None
        upper_bound = None
    else:
        max_ts = args.max_timestamp
        if args.min_timestamp is not None:
            lower_bound = args.min_timestamp
        elif args.days_from_max is not None:
            lower_bound = max_ts - args.days_from_max * 86400
        else:
            lower_bound = None

        if args.days_to_max is not None:
            upper_bound = max_ts - args.days_to_max * 86400
        else:
            upper_bound = None

    all_lengths, window_lengths, min_ts, max_ts_found = parse_idomaar_sessions(
        file_path, lower_bound=lower_bound, upper_bound=upper_bound
    )

    print("\n" + "=" * 80)
    print(" ZAKRES TIMESTAMPÓW W SUROWYM PLIKU IDOMAAR:")
    print("=" * 80)
    print(f" Min timestamp w zbiorze: {min_ts} ({format_ts(min_ts)})")
    print(f" Max timestamp w zbiorze: {max_ts_found} ({format_ts(max_ts_found)})")
    if lower_bound is not None or upper_bound is not None:
        print(f" Użyte okno czasowe:    {lower_bound} ({format_ts(lower_bound)}) do {upper_bound} ({format_ts(upper_bound)})")
    print("=" * 80)

    if not args.filter_only:
        analyze_lengths(all_lengths, f"Wszystkie surowe sesje ({os.path.basename(file_path)})")

    if not args.no_time_window and (lower_bound is not None or upper_bound is not None):
        lbl = f"Surowe sesje w oknie czasowym [{format_ts(lower_bound)} .. {format_ts(upper_bound)}] ({os.path.basename(file_path)})"
        analyze_lengths(window_lengths, lbl)


if __name__ == "__main__":
    main()
