#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_session_duration_distribution.py

Skrypt wyznaczający dokładny rozkład długości sesji (czasu trwania) w sekundach
dla zbiorów 30Music oraz LastFM-1K w wybranych oknach czasowych z preprocessingu:
- 30Music: domyślnie ostatnie 365 do 65 dni od MAX_TIMESTAMP (1421745720)
- LastFM-1K: domyślnie rok 2008 (1199145600 do 1230767999 UTC)

Pokazuje sesje 0s (1-elementowe), 1s, 2s, 3s, 4s, 5s, ..., 60s (1 minuta),
rozkład 61s-180s (1-3 minuty) oraz > 180s (> 3 minuty).
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime, timezone
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None


# --- Stałe konfiguracyjne z preprocessingu ---
_SESSION_INACTIVITY_GAP = 800  # Sekundy bezczynności dzielące na pod-sesje

# 30Music
MAX_TIMESTAMP_30MUSIC = 1421745720  # 2015-01-20 09:22:00 UTC
DEFAULT_30MUSIC_DAYS_FROM_MAX = 215
DEFAULT_30MUSIC_DAYS_TO_MAX = 65

# LastFM-1K (Rok 2008)
START_TIMESTAMP_LASTFM_2008 = 1199145600  # 2008-01-01 00:00:00 UTC
END_TIMESTAMP_LASTFM_2008 = 1230767999    # 2008-12-31 23:59:59 UTC

# Filtry nazwy artysty / utworu dla LastFM-1K
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
    """Zwraca True jeśli nazwa artysty lub utworu to szum / placeholder."""
    if not text or text.isspace() or '\ufffd' in text:
        return True
    if sum(1 for c in text if c.isalpha()) < 2:
        return True
    if _UNKNOWN_PATTERNS.search(text) or _URL_PATTERN.search(text):
        return True
    return False


def _parse_iso_timestamp(ts_str):
    """Konwertuje napis ISO 8601 na timestamp Unix UTC."""
    try:
        dt = datetime(
            int(ts_str[:4]), int(ts_str[5:7]), int(ts_str[8:10]),
            int(ts_str[11:13]), int(ts_str[14:16]), int(ts_str[17:19]),
            tzinfo=timezone.utc
        )
        return int(dt.timestamp())
    except Exception:
        return None


def _split_into_sub_sessions(tracks):
    """Dzieli listę utworów na pod-sesje przy przerwie > 800s."""
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


def extract_30music_durations(input_path, lower_bound, upper_bound):
    """Parsuje surowy plik 30Music (sessions.idomaar) i zwraca słownik ze statystykami sesji."""
    print(f"\n[30Music] Odczytywanie pliku: {input_path}")
    print(f"Okno czasowe: {lower_bound} .. {upper_bound} (Unix timestamp)")

    _PREFIX = "event.session\t"
    _PREFIX_LEN = len(_PREFIX)

    durations = []
    lengths = []
    in_window_count = 0

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Parsowanie 30Music"):
            if not line.startswith(_PREFIX):
                continue
            line_body = line[_PREFIX_LEN:]

            idx1 = line_body.find("\t")
            if idx1 == -1:
                continue
            idx2 = line_body.find("\t", idx1 + 1)
            if idx2 == -1:
                continue

            try:
                session_timestamp = int(line_body[idx1 + 1 : idx2])
            except ValueError:
                continue

            split_idx = line_body.find("} {")
            if split_idx == -1:
                continue

            try:
                session_objects = json.loads(line_body[split_idx + 2:].strip())
                objects = session_objects.get("objects", [])
                if not objects:
                    continue

                raw_tracks = []
                last_seen = {}
                for st in objects:
                    tid = st.get("id")
                    rel_ps = st.get("playstart", 0)
                    if tid is None:
                        continue

                    abs_ps = session_timestamp + rel_ps

                    if tid in last_seen and abs(abs_ps - last_seen[tid]) < 10:
                        continue
                    last_seen[tid] = abs_ps
                    raw_tracks.append({"id": tid, "ps": abs_ps})

                if not raw_tracks:
                    continue

                raw_tracks.sort(key=lambda x: x["ps"])
                sub_sess = _split_into_sub_sessions(raw_tracks)

                for sub in sub_sess:
                    playstarts = [t["ps"] for t in sub]
                    first_ts = playstarts[0] if playstarts else 0

                    if lower_bound <= first_ts <= upper_bound:
                        in_window_count += 1
                        dur = playstarts[-1] - playstarts[0] if len(playstarts) > 0 else 0
                        durations.append(dur)
                        lengths.append(len(sub))

            except (json.JSONDecodeError, ValueError, KeyError):
                continue

    return durations, lengths, in_window_count


def extract_lastfm_durations(input_path, lower_bound, upper_bound):
    """Parsuje surowy plik LastFM-1K i zwraca słownik ze statystykami sesji."""
    print(f"\n[LastFM-1K] Odczytywanie pliku: {input_path}")
    print(f"Okno czasowe: {lower_bound} .. {upper_bound} (Unix timestamp)")

    user_scrobbles = {}

    with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
        for line in tqdm(fin, desc="Odczytywanie scrobbli LastFM"):
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
            if not (lower_bound <= ts <= upper_bound):
                continue

            track_key = f"{artist_name}/_/{track_name}"

            if user_id not in user_scrobbles:
                user_scrobbles[user_id] = []
            user_scrobbles[user_id].append((ts, track_key))

    print(f"Wczytano scrobble dla {len(user_scrobbles):,} użytkowników.")

    durations = []
    lengths = []
    track_key_to_id = {}
    in_window_count = 0

    for user_id, scrobbles in tqdm(user_scrobbles.items(), desc="Budowanie pod-sesji LastFM"):
        scrobbles.sort(key=lambda x: x[0])

        dedup_tracks = []
        last_seen_ps = {}
        for ts, track_key in scrobbles:
            if track_key not in track_key_to_id:
                track_key_to_id[track_key] = len(track_key_to_id) + 1
            tid = track_key_to_id[track_key]

            if tid in last_seen_ps and abs(ts - last_seen_ps[tid]) < 10:
                continue
            last_seen_ps[tid] = ts
            dedup_tracks.append({"id": tid, "ps": ts})

        if not dedup_tracks:
            continue

        sub_sess = _split_into_sub_sessions(dedup_tracks)
        for sub in sub_sess:
            playstarts = [t["ps"] for t in sub]
            first_ts = playstarts[0] if playstarts else 0

            if lower_bound <= first_ts <= upper_bound:
                in_window_count += 1
                dur = playstarts[-1] - playstarts[0] if len(playstarts) > 0 else 0
                durations.append(dur)
                lengths.append(len(sub))

    return durations, lengths, in_window_count


def calculate_quantiles(data):
    """Wyznacza podstawowe statystyki opisowe i kwantyle dla podanej listy wartości."""
    if not data:
        return {}

    sorted_data = sorted(data)
    n = len(sorted_data)

    def percentile(p):
        k = (n - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < n else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    mean_val = sum(sorted_data) / n
    var_val = sum((x - mean_val) ** 2 for x in sorted_data) / n if n > 1 else 0.0

    return {
        "count": n,
        "mean": mean_val,
        "std": var_val ** 0.5,
        "min": sorted_data[0],
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": sorted_data[-1],
    }


def analyze_durations(durations, lengths):
    """Zlicza częstotliwości długości w sekundach i agreguje do zadanych przedziałów."""
    total = len(durations)
    if total == 0:
        return {}

    counts_exact = Counter(durations)
    single_item_count = sum(1 for l in lengths if l == 1)
    multi_item_count = sum(1 for l in lengths if l >= 2)

    # Przedziały sekundowe: 0s, 1s, 2s, 3s, 4s, 5s, ..., 60s
    per_second_0_to_60 = {}
    for s in range(61):
        c = counts_exact.get(s, 0)
        per_second_0_to_60[s] = {
            "count": c,
            "pct": (c / total) * 100.0 if total > 0 else 0.0
        }

    # Przedziały agregowane
    bins_definition = [
        ("0s (1 item)", lambda d: d == 0),
        ("1s - 5s", lambda d: 1 <= d <= 5),
        ("6s - 10s", lambda d: 6 <= d <= 10),
        ("11s - 15s", lambda d: 11 <= d <= 15),
        ("16s - 20s", lambda d: 16 <= d <= 20),
        ("21s - 29s", lambda d: 21 <= d <= 29),
        ("30s (filtr min)", lambda d: d == 30),
        ("31s - 45s", lambda d: 31 <= d <= 45),
        ("46s - 60s (1 min)", lambda d: 46 <= d <= 60),
        ("61s - 90s (1-1.5 min)", lambda d: 61 <= d <= 90),
        ("91s - 120s (1.5-2 min)", lambda d: 91 <= d <= 120),
        ("121s - 150s (2-2.5 min)", lambda d: 121 <= d <= 150),
        ("151s - 180s (2.5-3 min)", lambda d: 151 <= d <= 180),
        ("181s - 300s (3-5 min)", lambda d: 181 <= d <= 300),
        ("301s - 600s (5-10 min)", lambda d: 301 <= d <= 600),
        ("601s - 1800s (10-30 min)", lambda d: 601 <= d <= 1800),
        ("> 1800s (> 30 min)", lambda d: d > 1800),
    ]

    bin_results = []
    cum_pct = 0.0
    for label, cond in bins_definition:
        cnt = sum(c for d, c in counts_exact.items() if cond(d))
        pct = (cnt / total) * 100.0 if total > 0 else 0.0
        cum_pct += pct
        bin_results.append({
            "range": label,
            "count": cnt,
            "pct": pct,
            "cum_pct": cum_pct
        })

    # Podział po długości w elementach (k=1, k=2, k>2) i czasie trwania (<=30s vs >30s)
    len_eq_1 = sum(1 for l in lengths if l == 1)

    len_eq_2_total = sum(1 for l in lengths if l == 2)
    len_eq_2_le_30s = sum(1 for d, l in zip(durations, lengths) if l == 2 and d <= 30)
    len_eq_2_lt_30s = sum(1 for d, l in zip(durations, lengths) if l == 2 and d < 30)
    len_eq_2_gt_30s = sum(1 for d, l in zip(durations, lengths) if l == 2 and d > 30)

    len_gt_2_total = sum(1 for l in lengths if l > 2)
    len_gt_2_le_30s = sum(1 for d, l in zip(durations, lengths) if l > 2 and d <= 30)
    len_gt_2_lt_30s = sum(1 for d, l in zip(durations, lengths) if l > 2 and d < 30)
    len_gt_2_gt_30s = sum(1 for d, l in zip(durations, lengths) if l > 2 and d > 30)

    len_ge_2_le_30s = sum(1 for d, l in zip(durations, lengths) if l >= 2 and d <= 30)
    len_ge_2_lt_30s = sum(1 for d, l in zip(durations, lengths) if l >= 2 and d < 30)

    # Podsumowanie kluczowych progów
    lt_30s_cnt = sum(c for d, c in counts_exact.items() if d < 30)
    ge_30s_cnt = sum(c for d, c in counts_exact.items() if d >= 30)
    gt_180s_cnt = sum(c for d, c in counts_exact.items() if d > 180)
    gt_60s_cnt = sum(c for d, c in counts_exact.items() if d > 60)

    stats = calculate_quantiles(durations)

    return {
        "total_sessions": total,
        "single_item_sessions": single_item_count,
        "multi_item_sessions": multi_item_count,
        "len_eq_1": len_eq_1,
        "len_eq_1_pct": (len_eq_1 / total * 100) if total > 0 else 0.0,
        "len_eq_2_total": len_eq_2_total,
        "len_eq_2_total_pct": (len_eq_2_total / total * 100) if total > 0 else 0.0,
        "len_eq_2_le_30s": len_eq_2_le_30s,
        "len_eq_2_le_30s_pct": (len_eq_2_le_30s / total * 100) if total > 0 else 0.0,
        "len_eq_2_lt_30s": len_eq_2_lt_30s,
        "len_eq_2_lt_30s_pct": (len_eq_2_lt_30s / total * 100) if total > 0 else 0.0,
        "len_eq_2_gt_30s": len_eq_2_gt_30s,
        "len_eq_2_gt_30s_pct": (len_eq_2_gt_30s / total * 100) if total > 0 else 0.0,
        "len_gt_2_total": len_gt_2_total,
        "len_gt_2_total_pct": (len_gt_2_total / total * 100) if total > 0 else 0.0,
        "len_gt_2_le_30s": len_gt_2_le_30s,
        "len_gt_2_le_30s_pct": (len_gt_2_le_30s / total * 100) if total > 0 else 0.0,
        "len_gt_2_lt_30s": len_gt_2_lt_30s,
        "len_gt_2_lt_30s_pct": (len_gt_2_lt_30s / total * 100) if total > 0 else 0.0,
        "len_gt_2_gt_30s": len_gt_2_gt_30s,
        "len_gt_2_gt_30s_pct": (len_gt_2_gt_30s / total * 100) if total > 0 else 0.0,
        "len_ge_2_le_30s": len_ge_2_le_30s,
        "len_ge_2_le_30s_pct": (len_ge_2_le_30s / total * 100) if total > 0 else 0.0,
        "len_ge_2_lt_30s": len_ge_2_lt_30s,
        "len_ge_2_lt_30s_pct": (len_ge_2_lt_30s / total * 100) if total > 0 else 0.0,
        "lt_30s_count": lt_30s_cnt,
        "lt_30s_pct": (lt_30s_cnt / total * 100) if total > 0 else 0.0,
        "ge_30s_count": ge_30s_cnt,
        "ge_30s_pct": (ge_30s_count / total * 100) if total > 0 else 0.0,
        "gt_60s_count": gt_60s_cnt,
        "gt_60s_pct": (gt_60s_cnt / total * 100) if total > 0 else 0.0,
        "gt_180s_count": gt_180s_cnt,
        "gt_180s_pct": (gt_180s_cnt / total * 100) if total > 0 else 0.0,
        "per_second_0_to_60": per_second_0_to_60,
        "aggregated_bins": bin_results,
        "quantiles": stats
    }


def print_comparison_tables(res_30music, res_lastfm, window_30music_label, window_lastfm_label):
    """Drukuje czytelny raport z rozkładem i porównaniem obu zbiorów danych w konsoli."""

    print("\n" + "=" * 95)
    print("      ROZKŁAD DŁUGOŚCI SESJI W SEKUNDACH DLA 30MUSIC ORAZ LASTFM-1K")
    print("=" * 95)
    print(f"30Music Okno:   {window_30music_label}")
    print(f"LastFM-1K Okno: {window_lastfm_label}")
    print("-" * 95)

    n_30m = res_30music.get("total_sessions", 0) if res_30music else 0
    n_lfm = res_lastfm.get("total_sessions", 0) if res_lastfm else 0

    print(f"Łączna liczba pod-sesji w oknie:")
    print(f"  └─ 30Music:   {n_30m:>12,}")
    print(f"  └─ LastFM-1K: {n_lfm:>12,}")
    print("-" * 95)

    # 1. Tabela 0-60 sekund per-sekunda
    print("\n[TABELA 1] SZCZEGÓŁOWY ROZKŁAD DŁUGOŚCI SESJI (0s DO 60s, KAŻDA SEKUNDA):")
    print("-" * 95)
    print(f"{'Czas (s)':<10} | {'30Music Liczba':<16} | {'30Music %':<12} | {'LastFM Liczba':<16} | {'LastFM %':<12}")
    print("-" * 95)

    sec_30m = res_30music.get("per_second_0_to_60", {}) if res_30music else {}
    sec_lfm = res_lastfm.get("per_second_0_to_60", {}) if res_lastfm else {}

    for s in range(61):
        d_30 = sec_30m.get(s, {"count": 0, "pct": 0.0})
        d_lf = sec_lfm.get(s, {"count": 0, "pct": 0.0})
        lbl = f"{s}s" if s > 0 else "0s (1 item)"
        print(f"{lbl:<10} | {d_30['count']:>16,} | {d_30['pct']:>11.3f}% | {d_lf['count']:>16,} | {d_lf['pct']:>11.3f}%")

    print("-" * 95)

    # 2. Tabela Agregowanych Przedziałów
    print("\n[TABELA 2] AGREGOWANE PRZEDZIAŁY CZASOWE (W TYM 1-3 MINUTY ORAZ >3 MINUTY):")
    print("-" * 95)
    print(f"{'Przedział Czasowy':<26} | {'30Music Liczba':<14} | {'30Music %':<10} | {'LastFM Liczba':<14} | {'LastFM %':<10}")
    print("-" * 95)

    bins_30 = res_30music.get("aggregated_bins", []) if res_30music else []
    bins_lf = res_lastfm.get("aggregated_bins", []) if res_lastfm else []

    bins_dict_lf = {b["range"]: b for b in bins_lf}

    for b3 in bins_30:
        r_name = b3["range"]
        blf = bins_dict_lf.get(r_name, {"count": 0, "pct": 0.0})
        print(f"{r_name:<26} | {b3['count']:>14,} | {b3['pct']:>9.2f}% | {blf['count']:>14,} | {blf['pct']:>9.2f}%")

    print("-" * 95)

    # 3. Tabela Statystyk Opisowych i Kwantyli
    print("\n[TABELA 3] STATYSTYKI OPISOWE CZASU TRWANIA SESJI (W SEKUNDACH):")
    print("-" * 95)
    print(f"{'Metryka / Kwantyl':<26} | {'30Music':<28} | {'LastFM-1K':<28}")
    print("-" * 95)

    q3 = res_30music.get("quantiles", {}) if res_30music else {}
    ql = res_lastfm.get("quantiles", {}) if res_lastfm else {}

    metrics = [
        ("Średnia (Mean)", "mean", ".1f"),
        ("Odchylenie std (Std)", "std", ".1f"),
        ("Minimum", "min", "d"),
        ("Mediana (P50)", "median", ".1f"),
        ("Kwantyl P75", "p75", ".1f"),
        ("Kwantyl P90", "p90", ".1f"),
        ("Kwantyl P95", "p95", ".1f"),
        ("Kwantyl P99", "p99", ".1f"),
        ("Maksimum", "max", "d"),
    ]

    for m_label, key, fmt in metrics:
        v3 = q3.get(key, 0)
        vl = ql.get(key, 0)

        v3_str = f"{v3:{fmt}}" if isinstance(v3, (int, float)) else str(v3)
        vl_str = f"{vl:{fmt}}" if isinstance(vl, (int, float)) else str(vl)

        print(f"{m_label:<26} | {v3_str:>28} | {vl_str:>28}")

    print("-" * 95)

    # 4. Tabela Szczegółowej Analizy Sesji o Długości = 2 Oraz > 2 I Czasie ≤ 30s
    print("\n[TABELA 4] PODZIAŁ SESJI O DŁUGOŚCI = 2 ORAZ > 2 DLA CZASU TRWANIA ≤ 30s (I < 30s):")
    print("-" * 95)
    print(f"{'Kategoria Sesji':<42} | {'30Music Liczba':<14} | {'30Music %':<9} | {'LastFM Liczba':<14} | {'LastFM %':<9}")
    print("-" * 95)

    def _get_val(res, key_cnt, key_pct):
        if not res:
            return 0, 0.0
        return res.get(key_cnt, 0), res.get(key_pct, 0.0)

    rows_tab4 = [
        ("Sesje 1-elementowe (długość = 1, dur = 0s)", "len_eq_1", "len_eq_1_pct"),
        ("Sesje o długości = 2 (Łącznie)", "len_eq_2_total", "len_eq_2_total_pct"),
        ("  └─ Długość = 2 oraz czas <= 30s", "len_eq_2_le_30s", "len_eq_2_le_30s_pct"),
        ("  └─ Długość = 2 oraz czas < 30s", "len_eq_2_lt_30s", "len_eq_2_lt_30s_pct"),
        ("  └─ Długość = 2 oraz czas > 30s", "len_eq_2_gt_30s", "len_eq_2_gt_30s_pct"),
        ("Sesje o długości > 2 (Łącznie)", "len_gt_2_total", "len_gt_2_total_pct"),
        ("  └─ Długość > 2 oraz czas <= 30s (ZAPYTANIE)", "len_gt_2_le_30s", "len_gt_2_le_30s_pct"),
        ("  └─ Długość > 2 oraz czas < 30s", "len_gt_2_lt_30s", "len_gt_2_lt_30s_pct"),
        ("  └─ Długość > 2 oraz czas > 30s", "len_gt_2_gt_30s", "len_gt_2_gt_30s_pct"),
        ("Sesje o długości >= 2 i czas <= 30s (Razem)", "len_ge_2_le_30s", "len_ge_2_le_30s_pct"),
        ("Sesje o długości >= 2 i czas < 30s (Razem)", "len_ge_2_lt_30s", "len_ge_2_lt_30s_pct"),
    ]

    for label, k_cnt, k_pct in rows_tab4:
        c3, p3 = _get_val(res_30music, k_cnt, k_pct)
        cl, pl = _get_val(res_lastfm, k_cnt, k_pct)
        print(f"{label:<42} | {c3:>14,} | {p3:>8.2f}% | {cl:>14,} | {pl:>8.2f}%")

    print("=" * 95)


def generate_plots(res_30music, res_lastfm, output_path="session_duration_distribution.png"):
    """Generuje wykres porównawczy rozkładu długości sesji i zapisuje go do pliku PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ Matplotlib nie jest zainstalowany. Pomijanie generowania wykresów.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Wykres 1: 0-60 sekund
    seconds = list(range(61))
    pct_30m_60 = [res_30music["per_second_0_to_60"][s]["pct"] for s in seconds] if res_30music else []
    pct_lfm_60 = [res_lastfm["per_second_0_to_60"][s]["pct"] for s in seconds] if res_lastfm else []

    axes[0].plot(seconds, pct_30m_60, label="30Music", color="#1f77b4", linewidth=2)
    axes[0].plot(seconds, pct_lfm_60, label="LastFM-1K", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Rozkład długości sesji (0 do 60 sekund)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Czas trwania sesji (sekundy)", fontsize=10)
    axes[0].set_ylabel("Udział sesji (%)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Wykres 2: Agregowane przedziały
    if res_30music and res_lastfm:
        bins = [b["range"] for b in res_30music["aggregated_bins"]]
        pcts_30 = [b["pct"] for b in res_30music["aggregated_bins"]]
        pcts_lf = [b["pct"] for b in res_lastfm["aggregated_bins"]]

        x = np.arange(len(bins)) if np is not None else list(range(len(bins)))
        width = 0.4

        axes[1].bar(x - width/2, pcts_30, width, label="30Music", color="#1f77b4")
        axes[1].bar(x + width/2, pcts_lf, width, label="LastFM-1K", color="#ff7f0e")
        axes[1].set_title("Udział sesji w przedziałach czasowych", fontsize=12, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(bins, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("Udział sesji (%)", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"📊 Wykres został pomyślnie zapisany w: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Skrypt wyznaczający rozkład długości sesji w sekundach dla 30Music oraz LastFM-1K."
    )
    parser.add_argument("--path_30music", type=str, default="dataset_raw/sessions.idomaar",
                        help="Ścieżka do surowego pliku 30Music (default: dataset_raw/sessions.idomaar)")
    parser.add_argument("--path_lastfm", type=str, default="dataset_raw/userid-timestamp-artid-artname-traid-traname.tsv",
                        help="Ścieżka do surowego pliku LastFM-1K (default: dataset_raw/userid-timestamp-artid-artname-traid-traname.tsv)")
    parser.add_argument("--days_from_max", type=int, default=DEFAULT_30MUSIC_DAYS_FROM_MAX,
                        help=f"30Music: Dolna granica dni od MAX_TIMESTAMP (default: {DEFAULT_30MUSIC_DAYS_FROM_MAX})")
    parser.add_argument("--days_to_max", type=int, default=DEFAULT_30MUSIC_DAYS_TO_MAX,
                        help=f"30Music: Górna granica dni od MAX_TIMESTAMP (default: {DEFAULT_30MUSIC_DAYS_TO_MAX})")
    parser.add_argument("--save_json", type=str, default="session_duration_distribution.json",
                        help="Ścieżka do zapisu wyniku JSON (opcjonalne)")
    parser.add_argument("--save_csv", type=str, default=None,
                        help="Ścieżka do zapisu wyniku CSV (opcjonalne)")
    parser.add_argument("--plot", action="store_true",
                        help="Wygeneruj i zapisz wykres w formacie PNG")
    parser.add_argument("--plot_path", type=str, default="session_duration_distribution.png",
                        help="Ścieżka do pliku PNG wykresu")

    args = parser.parse_args()

    # Okno dla 30Music (od days_from_max do days_to_max)
    lower_30m = MAX_TIMESTAMP_30MUSIC - args.days_from_max * 86400
    upper_30m = MAX_TIMESTAMP_30MUSIC - args.days_to_max * 86400
    lbl_30m = f"Ostatnie {args.days_from_max} do {args.days_to_max} dni od MAX_TIMESTAMP (okno 150 dni z process.sh)"

    # Okno dla LastFM-1K (wyłącznie rok 2008)
    lower_lfm = START_TIMESTAMP_LASTFM_2008
    upper_lfm = END_TIMESTAMP_LASTFM_2008
    lbl_lfm = "Rok 2008 (2008-01-01 do 2008-12-31 UTC)"

    res_30m = None
    res_lfm = None

    if os.path.exists(args.path_30music):
        dur_30m, len_30m, _ = extract_30music_durations(args.path_30music, lower_30m, upper_30m)
        res_30m = analyze_durations(dur_30m, len_30m)
    else:
        print(f"⚠️ Plik 30Music nie istnieje: {args.path_30music}")

    if os.path.exists(args.path_lastfm):
        dur_lfm, len_lfm, _ = extract_lastfm_durations(args.path_lastfm, lower_lfm, upper_lfm)
        res_lfm = analyze_durations(dur_lfm, len_lfm)
    else:
        print(f"⚠️ Plik LastFM-1K nie istnieje: {args.path_lastfm}")

    # Wydruk tabel konsolowych
    print_comparison_tables(res_30m, res_lfm, lbl_30m, lbl_lfm)

    # Zapis wyników do JSON
    output_data = {
        "30music": {
            "window_label": lbl_30m,
            "lower_bound": lower_30m,
            "upper_bound": upper_30m,
            "results": res_30m
        },
        "lastfm": {
            "window_label": lbl_lfm,
            "lower_bound": lower_lfm,
            "upper_bound": upper_lfm,
            "results": res_lfm
        }
    }

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as fjson:
            json.dump(output_data, fjson, indent=2, ensure_ascii=False)
        print(f"💾 Wyniki zostały zapisane do pliku JSON: {args.save_json}")

    # Zapis wyników do CSV (jeśli zażądano)
    if args.save_csv:
        rows = []
        sec_30 = res_30m.get("per_second_0_to_60", {}) if res_30m else {}
        sec_lf = res_lfm.get("per_second_0_to_60", {}) if res_lfm else {}
        for s in range(61):
            d30 = sec_30.get(s, {"count": 0, "pct": 0.0})
            dlf = sec_lf.get(s, {"count": 0, "pct": 0.0})
            rows.append({
                "second": s,
                "30music_count": d30["count"],
                "30music_pct": d30["pct"],
                "lastfm_count": dlf["count"],
                "lastfm_pct": dlf["pct"],
            })
        if pd is not None:
            df = pd.DataFrame(rows)
            df.to_csv(args.save_csv, index=False)
            print(f"💾 Wyniki per-sekunda zostały zapisane do pliku CSV: {args.save_csv}")
        else:
            import csv
            with open(args.save_csv, "w", newline="", encoding="utf-8") as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=["second", "30music_count", "30music_pct", "lastfm_count", "lastfm_pct"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"💾 Wyniki per-sekunda zostały zapisane do pliku CSV: {args.save_csv}")

    if args.plot:
        generate_plots(res_30m, res_lfm, args.plot_path)


if __name__ == "__main__":
    main()
