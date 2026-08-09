#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_filtered_sessions.py

Skrypt i funkcja do dokładnego wyznaczania liczby sesji odpadających z SUROWYCH zbiorów
30Music oraz LastFM-1K w określonym oknie czasowym:
1. Sesje o długości > 100.
2. Sesje o długości < 2 (1-elementowe).
3. Sesje o czasie trwania < 60 sekund i długości >= 2 (wieloelementowe, lecz zbyt krótkie).

Przetwarzanie odbywa się bezpośrednio z surowych plików źródłowych:
- 30Music: dataset_raw/sessions.idomaar (tworzenie pod-sesji z przerwą > 800s)
- LastFM-1K: dataset_raw/userid-timestamp-artid-artname-traid-traname.tsv (grupowanie po użytkowniku, przerwa > 800s)
"""

import os
import re
import json
from datetime import datetime, timezone
from tqdm import tqdm

_SESSION_INACTIVITY_GAP = 800  # Przerwa nieaktywności dzieląca na pod-sesje (sekundy)

# 30Music
MAX_TIMESTAMP_30MUSIC = 1421745720  # 2015-01-20 09:22:00 UTC

# LastFM-1K (Rok 2008)
START_TIMESTAMP_LASTFM_2008 = 1199145600  # 2008-01-01 00:00:00 UTC
END_TIMESTAMP_LASTFM_2008 = 1230767999    # 2008-12-31 23:59:59 UTC

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
    """Zwraca True jeśli nazwa artysty/utworu jest szumem lub symbolem placeholder."""
    if not text or text.isspace() or '\ufffd' in text:
        return True
    if sum(1 for c in text if c.isalpha()) < 2:
        return True
    if _UNKNOWN_PATTERNS.search(text) or _URL_PATTERN.search(text):
        return True
    return False


def _parse_iso_timestamp(ts_str):
    """Konwertuje tekstowy timestamp ISO 8601 na timestamp Unix."""
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
    """Dzieli chronologiczną listę utworów na pod-sesje przy przerwie nieaktywności > 800s."""
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


def evaluate_sub_sessions(sub_sessions, lower_bound=None, upper_bound=None,
                           max_length=100, min_duration=60, min_length=2):
    """Ewaluuje utworzone surowe pod-sesje pod kątem okna czasowego oraz filtrów długości/czasu.

    Rozdzielenie na wzajemnie rozłączne kategorie:
    - Sesje o długości > 100
    - Sesje o długości < 2 (1-elementowe)
    - Sesje o czasie trwania < 60s i długości >= 2 (wieloelementowe, lecz zbyt krótkie)
    - Sesje prawidłowe (2 <= długość <= 100 oraz czas trwania >= 60s)
    """
    total_raw_sub_sessions = len(sub_sessions)
    total_in_window = 0

    dropped_gt_max_len = 0
    dropped_lt_min_len = 0
    dropped_lt_60s_ge_2 = 0
    dropped_lt_60s_total = 0
    valid_sessions = 0

    for sub in sub_sessions:
        playstarts = [t["ps"] for t in sub]
        first_ts = playstarts[0] if playstarts else 0

        # Filtrowanie oknem czasowym
        if lower_bound is not None and first_ts < lower_bound:
            continue
        if upper_bound is not None and first_ts > upper_bound:
            continue

        total_in_window += 1

        length = len(sub)
        duration = (playstarts[-1] - playstarts[0]) if len(playstarts) > 0 else 0

        if duration < min_duration:
            dropped_lt_60s_total += 1

        if length > max_length:
            dropped_gt_max_len += 1
        elif length < min_length:
            dropped_lt_min_len += 1
        elif duration < min_duration:
            # Długość jest >= 2 oraz <= 100, ale czas trwania < 60 sekund
            dropped_lt_60s_ge_2 += 1
        else:
            valid_sessions += 1

    pct_gt_max_len = (dropped_gt_max_len / total_in_window * 100) if total_in_window > 0 else 0.0
    pct_lt_min_len = (dropped_lt_min_len / total_in_window * 100) if total_in_window > 0 else 0.0
    pct_lt_60s_ge_2 = (dropped_lt_60s_ge_2 / total_in_window * 100) if total_in_window > 0 else 0.0
    pct_lt_60s_total = (dropped_lt_60s_total / total_in_window * 100) if total_in_window > 0 else 0.0
    pct_valid = (valid_sessions / total_in_window * 100) if total_in_window > 0 else 0.0

    return {
        "total_raw_sub_sessions": total_raw_sub_sessions,
        "total_in_window": total_in_window,
        "dropped_gt_100": dropped_gt_max_len,
        "pct_gt_100": pct_gt_max_len,
        "dropped_lt_2_items": dropped_lt_min_len,
        "pct_lt_2_items": pct_lt_min_len,
        "dropped_lt_60s_ge_2": dropped_lt_60s_ge_2,
        "pct_lt_60s_ge_2": pct_lt_60s_ge_2,
        "dropped_lt_60s_total": dropped_lt_60s_total,
        "pct_lt_60s_total": pct_lt_60s_total,
        "valid_sessions": valid_sessions,
        "pct_valid": pct_valid,
    }


def analyze_30music_raw(days=150, days_from_max=None, days_to_max=None,
                        input_path="dataset_raw/sessions.idomaar"):
    """Parsuje surowy plik sessions.idomaar zbioru 30Music, buduje pod-sesje i zlicza odrzucenia w oknie czasowym."""
    if not os.path.exists(input_path):
        print(f"❌ Nie znaleziono pliku 30Music: {input_path}")
        return None

    if days_from_max is not None and days_to_max is not None:
        lower_bound = MAX_TIMESTAMP_30MUSIC - days_from_max * 86400
        upper_bound = MAX_TIMESTAMP_30MUSIC - days_to_max * 86400
        label_window = f"ostatnie {days_from_max} do {days_to_max} dni od MAX_TIMESTAMP"
    else:
        lower_bound = MAX_TIMESTAMP_30MUSIC - days * 86400
        upper_bound = MAX_TIMESTAMP_30MUSIC
        label_window = f"{days} dni (od MAX_TIMESTAMP)"

    print(f"\n[30Music] Parsowanie surowego pliku {input_path} (Okno: {label_window})...")
    all_sub_sessions = []

    _PREFIX = "event.session\t"
    _PREFIX_LEN = len(_PREFIX)

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Odczytywanie idomaar"):
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
                all_sub_sessions.extend(sub_sess)

            except (json.JSONDecodeError, ValueError, KeyError):
                continue

    res = evaluate_sub_sessions(
        all_sub_sessions,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        max_length=100,
        min_duration=60,
        min_length=2
    )

    _print_report(f"30Music (Surowy zbiór, {label_window})", input_path, res)
    return res


def analyze_lastfm_raw(input_path="dataset_raw/userid-timestamp-artid-artname-traid-traname.tsv"):
    """Parsuje surowy plik tsv LastFM-1K, grupuje scrobble użytkowników, buduje pod-sesje i zlicza odrzucenia w oknie rocznym (2008)."""
    if not os.path.exists(input_path):
        print(f"❌ Nie znaleziono pliku LastFM-1K: {input_path}")
        return None

    lower_bound = START_TIMESTAMP_LASTFM_2008
    upper_bound = END_TIMESTAMP_LASTFM_2008

    print(f"\n[LastFM-1K] Parsowanie surowego pliku {input_path} (Rok 2008)...")
    user_scrobbles = {}

    with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
        for line in tqdm(fin, desc="Odczytywanie scrobbli"):
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

    print(f"Wczytano scrobble dla {len(user_scrobbles):,} użytkowników w oknie 2008.")

    track_key_to_id = {}
    all_sub_sessions = []

    for user_id, scrobbles in tqdm(user_scrobbles.items(), desc="Budowanie pod-sesji"):
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
        all_sub_sessions.extend(sub_sess)

    res = evaluate_sub_sessions(
        all_sub_sessions,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        max_length=100,
        min_duration=60,
        min_length=2
    )

    _print_report("LastFM-1K (Surowy zbiór, Rok 2008)", input_path, res)
    return res


def _print_report(dataset_label, file_path, res):
    """Formatowany i przejrzysty wydruk wyników analizy."""
    print("\n" + "=" * 80)
    print(f"  ANALIZA ODPADAJĄCYCH SESJI Z SUROWEGO ZBIORU: {dataset_label}")
    print("=" * 80)
    print(f"Plik źródłowy:                             {file_path}")
    print(f"Łącznie utworzone surowe pod-sesje:        {res['total_raw_sub_sessions']:>12,}")
    print(f"Pod-sesje w wybranym oknie czasowym:       {res['total_in_window']:>12,}")
    print("-" * 80)
    print(f"❌ Sesje o długości > 100:                  {res['dropped_gt_100']:>12,} ({res['pct_gt_100']:>6.2f}%)")
    print(f"❌ Sesje o długości < 2 (1-elementowe):     {res['dropped_lt_2_items']:>12,} ({res['pct_lt_2_items']:>6.2f}%)")
    print(f"❌ Sesje < 60s przy długości ≥ 2:           {res['dropped_lt_60s_ge_2']:>12,} ({res['pct_lt_60s_ge_2']:>6.2f}%)")
    print(f"   (Łącznie wszystkich sesji < 60s:        {res['dropped_lt_60s_total']:>12,} [{res['pct_lt_60s_total']:>6.2f}%])")
    print("-" * 80)
    print(f"✅ Sesje prawidłowe (długość 2-100 i czas ≥ 60s): {res['valid_sessions']:>12,} ({res['pct_valid']:>6.2f}%)")
    print("=" * 80)


def run_analysis():
    """Główna funkcja uruchamiająca analizę dla obu surowych zbiorów."""
    print("Uruchamianie analizy odrzucania sesji z surowych danych...")

    # 1. 30Music z surowego pliku sessions.idomaar (okno 150 dni oraz opcjonalnie 365 do 65 dni)
    file_30music = "dataset_raw/sessions.idomaar"
    if os.path.exists(file_30music):
        analyze_30music_raw(days=150, input_path=file_30music)
        analyze_30music_raw(days_from_max=365, days_to_max=65, input_path=file_30music)
    else:
        print(f"⚠️ Nie znaleziono {file_30music}")

    # 2. LastFM-1K z surowego pliku userid-timestamp-artid-artname-traid-traname.tsv
    file_lastfm = "dataset_raw/userid-timestamp-artid-artname-traid-traname.tsv"
    if os.path.exists(file_lastfm):
        analyze_lastfm_raw(input_path=file_lastfm)
    else:
        print(f"⚠️ Nie znaleziono {file_lastfm}")


if __name__ == "__main__":
    run_analysis()
