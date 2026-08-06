#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt do analizy wpływu filtrowania utworów (track playcount threshold) na rozkład długości sesji
w surowym zbiorze 30Music (sessions.idomaar / sessions.tsv).

Porównuje 3 warianty:
1. Surowy zbiór bez filtrowania utworów (Raw / Min 1)
2. Usunięcie utworów z < 5 odtworzeniami w całym zbiorze (Min 5)
3. Usunięcie utworów z < 25 odtworzeniami w całym zbiorze (Min 25)
"""

import os
import sys
import json
import argparse
from collections import Counter
import numpy as np
from tqdm import tqdm

try:
    import orjson as fast_json
except ImportError:
    fast_json = json


def load_raw_sessions(file_path):
    """
    Ładuje surowe sesje z pliku sessions.tsv lub sessions.idomaar.
    Zwraca listę sesji, gdzie każda sesja to lista ID utworów: [[track_id1, track_id2, ...], ...]
    """
    sessions = []
    print(f"Odczytywanie surowych sesji z pliku: {file_path}...")
    
    fname = os.path.basename(file_path)
    
    if fname.endswith(".tsv"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in tqdm(f, desc="Odczyt sessions.tsv"):
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    try:
                        tracks_objs = fast_json.loads(parts[3])
                        track_ids = [t["id"] for t in tracks_objs if "id" in t]
                        if track_ids:
                            sessions.append(track_ids)
                    except Exception:
                        continue
    else:
        # Format .idomaar
        _PREFIX = "event.session\t"
        _PREFIX_LEN = len(_PREFIX)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in tqdm(f, desc="Odczyt sessions.idomaar"):
                if not line.startswith(_PREFIX):
                    continue
                line_body = line[_PREFIX_LEN:]
                split_idx = line_body.find("} {")
                if split_idx == -1:
                    continue
                try:
                    objects_json = line_body[split_idx + 2:].strip()
                    session_objects = fast_json.loads(objects_json)
                    tracks = session_objects.get("objects", [])
                    track_ids = [t["id"] for t in tracks if "id" in t]
                    if track_ids:
                        sessions.append(track_ids)
                except Exception:
                    continue

    print(f"Załadowano łącznie {len(sessions):,} surowych sesji.")
    return sessions


def analyze_filtering_impact(sessions):
    """
    Przeprowadza porównawczą analizę wpływu filtrowania utworów (Min 1 vs Min 5 vs Min 25).
    """
    print("\nObliczanie globalnej popularności utworów (playcounts)...")
    track_counts = Counter()
    for s in tqdm(sessions, desc="Zliczanie odtworzeń utworów"):
        track_counts.update(s)

    total_unique_tracks = len(track_counts)
    tracks_min5 = {tid for tid, count in track_counts.items() if count >= 5}
    tracks_min25 = {tid for tid, count in track_counts.items() if count >= 25}

    print("\n" + "=" * 80)
    print(" PODSUMOWANIE UNIKALNYCH UTWORÓW (TRACK PLAYCOUNT)")
    print("=" * 80)
    print(f"Wszystkie unikalne utwory (Min 1):  {total_unique_tracks:>12,}")
    print(f"Utwory z odtworzeniami >= 5:       {len(tracks_min5):>12,} (Usunięto {total_unique_tracks - len(tracks_min5):,} / {(1 - len(tracks_min5)/total_unique_tracks)*100:.2f}%)")
    print(f"Utwory z odtworzeniami >= 25:      {len(tracks_min25):>12,} (Usunięto {total_unique_tracks - len(tracks_min25):,} / {(1 - len(tracks_min25)/total_unique_tracks)*100:.2f}%)")
    print("-" * 80)

    # Obliczanie długości sesji dla każdego z wariantów
    lengths_raw = []
    lengths_min5 = []
    lengths_min25 = []

    print("\nFiltrowanie sesji według progów popularności utworów...")
    for s in tqdm(sessions, desc="Analiza długości po filtrowaniu"):
        l_raw = len(s)
        l_min5 = sum(1 for tid in s if tid in tracks_min5)
        l_min25 = sum(1 for tid in s if tid in tracks_min25)

        lengths_raw.append(l_raw)
        lengths_min5.append(l_min5)
        lengths_min25.append(l_min25)

    arr_raw = np.array(lengths_raw, dtype=np.int64)
    arr_min5 = np.array(lengths_min5, dtype=np.int64)
    arr_min25 = np.array(lengths_min25, dtype=np.int64)

    total_sess = len(arr_raw)

    # PORÓWNANIE METRYK GŁÓWNYCH
    print("\n" + "=" * 80)
    print(" PORÓWNANIE OGÓLNE (OGÓLNE STATYSTYKI ZBIORU)")
    print("=" * 80)
    print(f" {'Metryka':<32} | {'Surowy (Min 1)':>14} | {'Filtrowany (Min 5)':>18} | {'Filtrowany (Min 25)':>18}")
    print("-" * 88)
    print(f" {'Łączna liczba zdarzeń (utworów)':<32} | {arr_raw.sum():>14,} | {arr_min5.sum():>18,} | {arr_min25.sum():>18,}")
    print(f" {'Odsetek zachowanych zdarzeń':<32} | {'100.00%':>14} | {(arr_min5.sum()/arr_raw.sum())*100:>17.2f}% | {(arr_min25.sum()/arr_raw.sum())*100:>17.2f}%")
    print(f" {'Średnia długość sesji':<32} | {arr_raw.mean():>14.2f} | {arr_min5.mean():>18.2f} | {arr_min25.mean():>18.2f}")
    print(f" {'Mediana długości sesji':<32} | {np.median(arr_raw):>14.0f} | {np.median(arr_min5):>18.0f} | {np.median(arr_min25):>18.0f}")
    print(f" {'Odchylenie standardowe':<32} | {arr_raw.std():>14.2f} | {arr_min5.std():>18.2f} | {arr_min25.std():>18.2f}")
    print(f" {'Sesje puste (długość == 0)':<32} | {(arr_raw==0).sum():>14,} | {(arr_min5==0).sum():>18,} | {(arr_min25==0).sum():>18,}")
    print(f" {'Sesje o długości == 1':<32} | {(arr_raw==1).sum():>14,} | {(arr_min5==1).sum():>18,} | {(arr_min25==1).sum():>18,}")
    print(f" {'Prawidłowe sesje (długość >= 2)':<32} | {(arr_raw>=2).sum():>14,} | {(arr_min5>=2).sum():>18,} | {(arr_min25>=2).sum():>18,}")
    print(f" {'% zachowanych sesji (len >= 2)':<32} | {(arr_raw>=2).mean()*100:>13.2f}% | {(arr_min5>=2).mean()*100:>17.2f}% | {(arr_min25>=2).mean()*100:>17.2f}%")
    print("-" * 88)

    # DOKŁADNY ROZKŁAD DŁUGOŚCI SESJI (OD 1 DO 10 ORAZ PRZEDZIAŁY)
    print("\n" + "=" * 80)
    print(" DOKŁADNY ROZKŁAD DŁUGOŚCI SESJI (LICZBA SESJI / ODSOTEK %)")
    print("=" * 80)
    print(f" {'Długość sesji':<16} | {'Surowy (Min 1)':>20} | {'Filtrowany (Min 5)':>22} | {'Filtrowany (Min 25)':>22}")
    print("-" * 86)

    c_raw = Counter(arr_raw)
    c_min5 = Counter(arr_min5)
    c_min25 = Counter(arr_min25)

    for l in range(0, 11):
        r_cnt, r_pct = c_raw[l], (c_raw[l] / total_sess) * 100
        m5_cnt, m5_pct = c_min5[l], (c_min5[l] / total_sess) * 100
        m25_cnt, m25_pct = c_min25[l], (c_min25[l] / total_sess) * 100
        
        lbl = f"Długość == {l}"
        print(f" {lbl:<16} | {r_cnt:>10,} ({r_pct:>5.1f}%) | {m5_cnt:>12,} ({m5_pct:>5.1f}%) | {m25_cnt:>12,} ({m25_pct:>5.1f}%)")

    # Przedziały
    ranges = [(11, 15), (16, 20), (21, 50), (51, 100)]
    for start, end in ranges:
        r_cnt = sum(c for k, c in c_raw.items() if start <= k <= end)
        m5_cnt = sum(c for k, c in c_min5.items() if start <= k <= end)
        m25_cnt = sum(c for k, c in c_min25.items() if start <= k <= end)

        r_pct = (r_cnt / total_sess) * 100
        m5_pct = (m5_cnt / total_sess) * 100
        m25_pct = (m25_cnt / total_sess) * 100

        lbl = f"Długość {start}-{end}"
        print(f" {lbl:<16} | {r_cnt:>10,} ({r_pct:>5.1f}%) | {m5_cnt:>12,} ({m5_pct:>5.1f}%) | {m25_cnt:>12,} ({m25_pct:>5.1f}%)")

    # Długość > 100
    r_gt = sum(c for k, c in c_raw.items() if k > 100)
    m5_gt = sum(c for k, c in c_min5.items() if k > 100)
    m25_gt = sum(c for k, c in c_min25.items() if k > 100)

    print(f" {'Długość > 100':<16} | {r_gt:>10,} ({(r_gt/total_sess)*100:>5.1f}%) | {m5_gt:>12,} ({(m5_gt/total_sess)*100:>5.1f}%) | {m25_gt:>12,} ({(m25_gt/total_sess)*100:>5.1f}%)")
    print("-" * 86)

    # SKUMULOWANE STATYSTYKI SESJI (<= X)
    print("\n" + "=" * 80)
    print(" SKUMULOWANY ROZKŁAD DŁUGOŚCI SESJI (DŁUGOŚĆ <= X)")
    print("=" * 80)
    print(f" {'Długość <= X':<16} | {'Surowy (Min 1)':>20} | {'Filtrowany (Min 5)':>22} | {'Filtrowany (Min 25)':>22}")
    print("-" * 86)

    cum_thresholds = [1, 2, 3, 4, 5, 10, 15, 20, 50, 100]
    for th in cum_thresholds:
        r_le = sum(c for k, c in c_raw.items() if k <= th)
        m5_le = sum(c for k, c in c_min5.items() if k <= th)
        m25_le = sum(c for k, c in c_min25.items() if k <= th)

        r_pct = (r_le / total_sess) * 100
        m5_pct = (m5_le / total_sess) * 100
        m25_pct = (m25_le / total_sess) * 100

        lbl = f"Długość <= {th}"
        print(f" {lbl:<16} | {r_le:>10,} ({r_pct:>5.1f}%) | {m5_le:>12,} ({m5_pct:>5.1f}%) | {m25_le:>12,} ({m25_pct:>5.1f}%)")

    print("-" * 86)

    # PORÓWNANIE PERCENTYLI
    print("\n" + "=" * 80)
    print(" PORÓWNANIE PERCENTYLI DŁUGOŚCI SESJI")
    print("=" * 80)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9, 100]
    
    p_raw = np.percentile(arr_raw, percentiles)
    p_m5 = np.percentile(arr_min5, percentiles)
    p_m25 = np.percentile(arr_min25, percentiles)

    print(f" {'Percentyl':<16} | {'Surowy (Min 1)':>16} | {'Filtrowany (Min 5)':>20} | {'Filtrowany (Min 25)':>20}")
    print("-" * 78)
    for p, v_r, v_m5, v_m25 in zip(percentiles, p_raw, p_m5, p_m25):
        lbl = f"P{p}" if p not in (25, 50, 75, 100) else f"P{p} ({'Q1' if p==25 else 'Mediana' if p==50 else 'Q3' if p==75 else 'Max'})"
        print(f" {lbl:<16} | {v_r:>16.1f} | {v_m5:>20.1f} | {v_m25:>20.1f}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analiza wpływu usunięcia rzadkich utworów (Min 5 vs Min 25) na surowy zbiór 30Music."
    )
    parser.add_argument(
        "--file", "-f", type=str, default="dataset_raw/sessions.tsv",
        help="Ścieżka do surowego pliku sessions.tsv lub sessions.idomaar."
    )

    args = parser.parse_args()
    file_path = args.file

    if not os.path.exists(file_path):
        # Sprawdź alternatywy
        alt_tsv = os.path.join("dataset_raw", "sessions.tsv")
        alt_ido = os.path.join("dataset_raw", "sessions.idomaar")
        if os.path.exists(alt_tsv):
            file_path = alt_tsv
        elif os.path.exists(alt_ido):
            file_path = alt_ido
        else:
            print(f"[BŁĄD] Nie znaleziono pliku surowych sesji ({file_path}).")
            sys.exit(1)

    sessions = load_raw_sessions(file_path)
    analyze_filtering_impact(sessions)


if __name__ == "__main__":
    main()
