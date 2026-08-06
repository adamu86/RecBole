"""
Generate dataset statistics comparison table for LaTeX.

Computes metrics for both raw and processed versions of 30Music and LastFM-1k:
1. Liczba odsłuchań (interakcji)  – total individual listens/scrobbles
2. Liczba sesji                   – total sessions (after sub-session splitting)
3. Liczba unikalnych utworów      – unique track IDs
4. Liczba unikalnych artystów     – unique artist names
5. Liczba unikalnych użytkowników – unique user IDs
6. Średnia długość sesji          – mean tracks per session
7. Mediana długości sesji         – median tracks per session
8. Stopień rozrzedzenia (sparsity) – 1 - (unique_user_item_pairs / (users * items))

Processing pipelines exactly mirror generate_raw_plots.py (raw) and
generate_processed_plots.py (processed).

Output: prints stats to stdout AND writes documents/dataset_stats_comparison.tex
"""

import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import numpy as np
from tqdm import tqdm

try:
    import orjson as fast_json
except ImportError:
    fast_json = json

# ───────────────────────────── paths ─────────────────────────────
RAW_DATA_DIR = Path("dataset_raw")
DOCUMENTS_DIR = Path("documents")

RAW_30MUSIC_FILE = RAW_DATA_DIR / "sessions.idomaar"
RAW_LASTFM_FILE = RAW_DATA_DIR / "userid-timestamp-artid-artname-traid-traname.tsv"

TRACKS_30MUSIC_FILE = RAW_DATA_DIR / "tracks.tsv"
TRACKS_LASTFM_FILE = RAW_DATA_DIR / "lastfm_tracks.tsv"

# ───────────────────────────── constants ─────────────────────────
MIN_TRACK_PLAYCOUNT = 25
MIN_SESSION_LENGTH = 2
MAX_SESSION_LENGTH = 100
MIN_SESSION_PLAYTIME = 30
MAX_SESSION_PLAYTIME = 1_000_000
SESSION_GAP = 800

# LastFM 2-year time window
LASTFM_START_TS = 1167609600   # 2007-01-01 00:00:00 UTC
LASTFM_END_TS   = 1230767999   # 2008-12-31 23:59:59 UTC

# 30Music time window constants
MAX_TIMESTAMP_30M = 1421745720
DAYS_FROM_MAX_30M = 365
DAYS_TO_MAX_30M   = 65

# Noise filtering (LastFM)
_UNKNOWN_PATTERNS = re.compile(
    r'(?i)^(\[?unknown\]?|\[?none\]?|\[?null\]?|\[?deleted\]?|\[?untagged\]?'
    r'|n/a|na|none|null|\?+|<artista desconocido>|<nieznany wykonawca>)$'
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


def _split_into_sub_sessions(tracks, gap):
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


def load_30music_track_artist_map():
    """Build a mapping of track_id -> artist_name for 30Music."""
    track_to_artist = {}
    
    # 1. Load from dataset/artists.tsv and dataset_raw/tracks.tsv
    for path in [Path("dataset/artists.tsv"), TRACKS_30MUSIC_FILE]:
        if path.exists():
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        tid = parts[0].strip()
                        val = parts[1].strip()
                        artist = val.split("/_/")[0].strip() if "/_/" in val else val
                        if artist:
                            try:
                                track_to_artist[int(tid)] = artist
                            except ValueError:
                                pass
                            track_to_artist[tid] = artist

    # 2. Parse dataset_raw/tracks.idomaar for remaining tracks if present
    tracks_idomaar = RAW_DATA_DIR / "tracks.idomaar"
    if tracks_idomaar.exists():
        print(f"Supplementing 30Music artist map from {tracks_idomaar}...")
        with open(tracks_idomaar, "r", encoding="utf-8", errors="replace") as f:
            for line in tqdm(f, desc="Loading artists from tracks.idomaar"):
                if not line.startswith("track\t"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 4:
                    try:
                        tid = int(parts[1])
                        if tid in track_to_artist:
                            continue
                        name_idx = parts[3].find('"name":"')
                        if name_idx != -1:
                            end_idx = parts[3].find('"', name_idx + 8)
                            if end_idx != -1:
                                raw_name = parts[3][name_idx + 8 : end_idx]
                                unquoted = unquote_plus(raw_name)
                                if "/_/" in unquoted:
                                    artist = unquoted.split("/_/")[0].strip()
                                    if artist:
                                        track_to_artist[tid] = artist
                                        track_to_artist[str(tid)] = artist
                    except Exception:
                        pass

    print(f"Loaded artist map for {len(track_to_artist):,} 30Music tracks.")
    return track_to_artist


# ═══════════════════════════ RAW 30Music ═══════════════════════════
def compute_raw_30music():
    """Compute raw 30Music stats from sessions.idomaar (full timeline)."""
    print(f"\n[30Music RAW] Processing {RAW_30MUSIC_FILE}...")
    track_artist_map = load_30music_track_artist_map()

    total_interactions = 0
    session_lengths = []
    track_counter = Counter()
    artist_set = set()
    user_set = set()
    user_item_pairs = set()

    _PREFIX = "event.session\t"
    _PREFIX_LEN = len(_PREFIX)

    with open(RAW_30MUSIC_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Reading 30Music raw"):
            if not line.startswith(_PREFIX):
                continue
            body = line[_PREFIX_LEN:]
            try:
                idx1 = body.find("\t")
                if idx1 == -1:
                    continue
                idx2 = body.find("\t", idx1 + 1)
                if idx2 == -1:
                    continue

                ts = int(body[idx1 + 1 : idx2])
                if not (1388534400 <= ts <= 1425168000):
                    continue

                split_idx = body.find("} {")
                if split_idx == -1:
                    continue

                obj_str = body[split_idx + 2:].strip()
                session_objects = fast_json.loads(obj_str)
                tracks = session_objects.get("objects", [])
                if not tracks:
                    continue

                user_id = (session_objects.get("subjects", [{}])[0].get("id")
                           if session_objects.get("subjects") else None)
                if user_id is not None:
                    user_set.add(user_id)

                n_tracks = len(tracks)
                total_interactions += n_tracks
                session_lengths.append(n_tracks)

                for t in tracks:
                    tid = t.get("id")
                    if tid is not None:
                        track_counter[tid] += 1
                        if user_id is not None:
                            user_item_pairs.add((user_id, tid))
                        artist = track_artist_map.get(tid) or track_artist_map.get(str(tid))
                        if artist:
                            artist_set.add(artist)

            except Exception:
                continue

    session_lengths_arr = np.array(session_lengths)
    n_sessions = len(session_lengths)
    n_users = len(user_set)
    n_items = len(track_counter)
    n_artists = len(artist_set)
    mean_len = float(session_lengths_arr.mean()) if n_sessions > 0 else 0.0
    median_len = float(np.median(session_lengths_arr)) if n_sessions > 0 else 0.0
    n_unique_pairs = len(user_item_pairs)
    sparsity = 1.0 - (n_unique_pairs / (n_users * n_items)) if (n_users * n_items) > 0 else 1.0

    stats = {
        "interactions": total_interactions,
        "sessions": n_sessions,
        "unique_tracks": n_items,
        "unique_artists": n_artists,
        "unique_users": n_users,
        "mean_session_length": mean_len,
        "median_session_length": median_len,
        "sparsity": sparsity,
    }
    _print_stats("30Music (Surowy)", stats)
    return stats


# ═══════════════════════════ RAW LastFM-1k ═══════════════════════════
def compute_raw_lastfm():
    """Compute raw LastFM-1k stats from scrobbles TSV (full 2005-2009 timeline).

    Sessions are built using the same 800s inactivity gap, no filtering.
    """
    print(f"\n[LastFM-1k RAW] Processing {RAW_LASTFM_FILE}...")

    user_scrobbles = defaultdict(list)
    total_lines = sum(1 for _ in open(RAW_LASTFM_FILE, "rb"))

    with open(RAW_LASTFM_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, total=total_lines, desc="Reading LastFM raw"):
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
            user_scrobbles[user_id].append((ts, track_key, artist))

    # Build sessions
    total_interactions = 0
    session_lengths = []
    track_counter = Counter()
    artist_set = set()
    user_set = set()
    user_item_pairs = set()

    for user_id, scrobbles in tqdm(user_scrobbles.items(), desc="Building raw sessions"):
        scrobbles.sort(key=lambda x: x[0])
        if not scrobbles:
            continue

        # Split on gap
        sess_tracks = [(scrobbles[0][1], scrobbles[0][2])]
        prev_ts = scrobbles[0][0]

        for ts, t_key, art in scrobbles[1:]:
            if ts - prev_ts > SESSION_GAP:
                # Close previous session
                n = len(sess_tracks)
                session_lengths.append(n)
                total_interactions += n
                user_set.add(user_id)
                for tk, a in sess_tracks:
                    user_item_pairs.add((user_id, tk))
                    track_counter[tk] += 1
                    artist_set.add(a)
                sess_tracks = []

            sess_tracks.append((t_key, art))
            prev_ts = ts

        # Close last session
        if sess_tracks:
            n = len(sess_tracks)
            session_lengths.append(n)
            total_interactions += n
            user_set.add(user_id)
            for tk, a in sess_tracks:
                track_counter[tk] += 1
                artist_set.add(a)
                user_item_pairs.add((user_id, tk))

    session_lengths_arr = np.array(session_lengths)
    n_sessions = len(session_lengths)
    n_users = len(user_set)
    n_items = len(track_counter)
    n_artists = len(artist_set)
    mean_len = float(session_lengths_arr.mean()) if n_sessions > 0 else 0.0
    median_len = float(np.median(session_lengths_arr)) if n_sessions > 0 else 0.0
    n_unique_pairs = len(user_item_pairs)
    sparsity = 1.0 - (n_unique_pairs / (n_users * n_items)) if (n_users * n_items) > 0 else 1.0

    stats = {
        "interactions": total_interactions,
        "sessions": n_sessions,
        "unique_tracks": n_items,
        "unique_artists": n_artists,
        "unique_users": n_users,
        "mean_session_length": mean_len,
        "median_session_length": median_len,
        "sparsity": sparsity,
    }
    _print_stats("LastFM-1k (Surowy)", stats)
    return stats


# ═══════════════════════════ PROCESSED 30Music ═══════════════════════════
def _find_dataset_dirs(prefix):
    """Return sorted list of dataset dirs matching *prefix* inside dataset/."""
    dataset_base = Path("dataset")
    if not dataset_base.exists():
        return []
    return sorted([
        dataset_base / d for d in os.listdir(dataset_base)
        if (dataset_base / d).is_dir() and d.startswith(prefix)
    ])






def _compute_processed_from_inter(prefix, label):
    """Compute processed stats by reading .inter files from dataset directories.

    Reads the main .inter file (not train/valid/test splits) from each fold
    directory matching *prefix*, merges them via union for unique counts
    and sum for interactions, then computes session-level statistics.
    """
    dirs = _find_dataset_dirs(prefix)
    if not dirs:
        print(f"\n[{label}] No dataset directories found matching '{prefix}' in dataset/")
        return None

    print(f"\n[{label}] Reading .inter files from {len(dirs)} fold directories...")

    global_items = set()
    global_sessions = set()
    global_users = set()
    global_interactions = 0
    global_user_item_pairs = set()
    all_session_lengths = []  # lengths from counting rows per session across all folds

    for d in dirs:
        all_files = os.listdir(d)
        main_inters = [
            f for f in all_files
            if f.endswith(".inter")
            and not f.endswith((".train.inter", ".valid.inter", ".test.inter"))
        ]

        for fname in main_inters:
            fpath = d / fname
            print(f"  Parsing {fpath}...")

            session_row_counts = Counter()  # session_id -> row count
            items_local = set()
            users_local = set()
            session_user_local = {}  # session_id -> user_id

            with open(fpath, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                sep = "\t" if "\t" in header else " "
                cols = [c.split(":")[0] for c in header.split(sep)]
                item_idx = cols.index("item_id") if "item_id" in cols else -1
                session_idx = cols.index("session_id") if "session_id" in cols else -1
                user_idx = cols.index("user_id") if "user_id" in cols else -1

                if item_idx == -1:
                    print(f"    [OSTRZEŻENIE] Brak kolumny item_id, pomijam")
                    continue

                for line in tqdm(f, desc=f"    {fname}", leave=False):
                    parts = line.rstrip("\n").split(sep)
                    item = parts[item_idx]
                    items_local.add(item)
                    global_interactions += 1

                    if session_idx != -1:
                        sid = parts[session_idx]
                        session_row_counts[sid] += 1
                        global_sessions.add(sid)

                    if user_idx != -1:
                        uid = parts[user_idx]
                        users_local.add(uid)
                        if session_idx != -1:
                            session_user_local[sid] = uid

                    if user_idx != -1 and item_idx != -1:
                        global_user_item_pairs.add((parts[user_idx], item))

            global_items.update(items_local)
            global_users.update(users_local)

            # Session lengths from this fold
            for sid, cnt in session_row_counts.items():
                all_session_lengths.append(cnt)



    # Collect artists from tracks.tsv files
    artist_set = set()
    for d in dirs:
        tracks_tsv = d / "tracks.tsv"
        if tracks_tsv.exists():
            with open(tracks_tsv, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split("\t", 2)
                    if len(parts) >= 2:
                        name = parts[1].strip()
                        artist = name.split("/_/")[0].strip() if "/_/" in name else ""
                        if artist:
                            artist_set.add(artist)

    session_lengths_arr = np.array(all_session_lengths) if all_session_lengths else np.array([0])
    n_sessions = len(global_sessions)
    n_users = len(global_users)
    n_items = len(global_items)
    n_artists = len(artist_set)
    mean_len = float(session_lengths_arr.mean()) if len(all_session_lengths) > 0 else 0.0
    median_len = float(np.median(session_lengths_arr)) if len(all_session_lengths) > 0 else 0.0
    n_unique_pairs = len(global_user_item_pairs)
    sparsity = 1.0 - (n_unique_pairs / (n_users * n_items)) if (n_users * n_items) > 0 else 1.0

    stats = {
        "interactions": global_interactions,
        "sessions": n_sessions,
        "unique_tracks": n_items,
        "unique_artists": n_artists,
        "unique_users": n_users,
        "mean_session_length": mean_len,
        "median_session_length": median_len,
        "sparsity": sparsity,
    }
    _print_stats(label, stats)
    return stats


def compute_processed_30music():
    """Compute processed 30Music stats from actual .inter files in dataset/."""
    return _compute_processed_from_inter("30music", "30Music (Przetworzony)")


# ═══════════════════════════ PROCESSED LastFM-1k ═══════════════════════════
def compute_processed_lastfm():
    """Compute processed LastFM-1k stats from actual .inter files in dataset/."""
    return _compute_processed_from_inter("lastfm1k", "LastFM-1k (Przetworzony)")


# ═══════════════════════════ output helpers ═══════════════════════════
def _print_stats(label, stats):
    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  Interakcje:         {stats['interactions']:>14,}")
    print(f"  Sesje:              {stats['sessions']:>14,}")
    print(f"  Unikalne utwory:    {stats['unique_tracks']:>14,}")
    print(f"  Unikalni artyści:   {stats['unique_artists']:>14,}")
    print(f"  Unikalni użytk.:   {stats['unique_users']:>14,}")
    print(f"  Śr. dł. sesji:     {stats['mean_session_length']:>14.2f}")
    print(f"  Mediana dł. sesji: {stats['median_session_length']:>14.2f}")
    print(f"  Sparsity:          {stats['sparsity']*100:>13.2f}%")
    print(sep)


def _fmt_int(n):
    """Format integer with thin spaces (LaTeX-friendly)."""
    return f"{n:,}".replace(",", "\\,")


def _fmt_float(v, decimals=2):
    """Format float with comma as decimal separator (Polish convention)."""
    return f"{v:.{decimals}f}".replace(".", ",")


def _fmt_pct(v):
    """Format sparsity percentage."""
    return f"{v*100:.2f}\\%".replace(".", ",")


def write_latex_table(stats_30m_raw, stats_30m_proc, stats_lastfm_raw, stats_lastfm_proc,
                      output_path):
    """Write the LaTeX table to disk."""

    rows = [
        ("Liczba odsłuchań (interakcji)",
         _fmt_int(stats_30m_raw["interactions"]),
         _fmt_int(stats_30m_proc["interactions"]),
         _fmt_int(stats_lastfm_raw["interactions"]),
         _fmt_int(stats_lastfm_proc["interactions"])),

        ("Liczba sesji",
         _fmt_int(stats_30m_raw["sessions"]),
         _fmt_int(stats_30m_proc["sessions"]),
         _fmt_int(stats_lastfm_raw["sessions"]),
         _fmt_int(stats_lastfm_proc["sessions"])),

        ("Liczba unikalnych utworów",
         _fmt_int(stats_30m_raw["unique_tracks"]),
         _fmt_int(stats_30m_proc["unique_tracks"]),
         _fmt_int(stats_lastfm_raw["unique_tracks"]),
         _fmt_int(stats_lastfm_proc["unique_tracks"])),

        ("Liczba unikalnych artystów",
         _fmt_int(stats_30m_raw["unique_artists"]),
         _fmt_int(stats_30m_proc["unique_artists"]),
         _fmt_int(stats_lastfm_raw["unique_artists"]),
         _fmt_int(stats_lastfm_proc["unique_artists"])),

        ("Liczba unikalnych użytkowników",
         _fmt_int(stats_30m_raw["unique_users"]),
         _fmt_int(stats_30m_proc["unique_users"]),
         _fmt_int(stats_lastfm_raw["unique_users"]),
         _fmt_int(stats_lastfm_proc["unique_users"])),

        ("Średnia długość sesji",
         _fmt_float(stats_30m_raw["mean_session_length"]),
         _fmt_float(stats_30m_proc["mean_session_length"]),
         _fmt_float(stats_lastfm_raw["mean_session_length"]),
         _fmt_float(stats_lastfm_proc["mean_session_length"])),

        ("Mediana długości sesji",
         _fmt_float(stats_30m_raw["median_session_length"]),
         _fmt_float(stats_30m_proc["median_session_length"]),
         _fmt_float(stats_lastfm_raw["median_session_length"]),
         _fmt_float(stats_lastfm_proc["median_session_length"])),

        ("Stopień rozrzedzenia (\\textit{sparsity})",
         _fmt_pct(stats_30m_raw["sparsity"]),
         _fmt_pct(stats_30m_proc["sparsity"]),
         _fmt_pct(stats_lastfm_raw["sparsity"]),
         _fmt_pct(stats_lastfm_proc["sparsity"])),
    ]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Porównanie statystyk zbiorów 30Music oraz LastFM-1k przed i po przetworzeniu}")
    lines.append(r"\label{tab:dataset_stats_comparison}")
    lines.append(r"\begin{tabularx}{\textwidth}{l*{4}{>{\centering\arraybackslash}X}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Metryka / Parametr} & \textbf{30Music (Surowy)} & \textbf{30Music (Przetworzony)}"
        r" & \textbf{LastFM-1k (Surowy)} & \textbf{LastFM-1k (Przetworzony)} \\"
    )
    lines.append(r"\midrule")

    for label, *vals in rows:
        lines.append(f"{label} & {' & '.join(vals)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")
    lines.append("")  # trailing newline

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[OK] LaTeX table written to {output_path}")


def main():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    output_tex = DOCUMENTS_DIR / "dataset_stats_comparison.tex"

    # ─── Compute all four stat sets ───
    stats_30m_raw = compute_raw_30music()
    stats_lastfm_raw = compute_raw_lastfm()
    stats_30m_proc = compute_processed_30music()
    stats_lastfm_proc = compute_processed_lastfm()

    # ─── Summary table to stdout ───
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"{'PODSUMOWANIE STATYSTYK ZBIORÓW DANYCH':^100}")
    print(sep)
    header = (
        f"{'Metryka':<35} | {'30Music Raw':>14} | {'30Music Proc':>14}"
        f" | {'LastFM Raw':>14} | {'LastFM Proc':>14}"
    )
    print(header)
    print("─" * 100)

    metrics = [
        ("Interakcje",                  "interactions"),
        ("Sesje",                       "sessions"),
        ("Unikalne utwory",             "unique_tracks"),
        ("Unikalni artyści",            "unique_artists"),
        ("Unikalni użytkownicy",        "unique_users"),
        ("Średnia długość sesji",       "mean_session_length"),
        ("Mediana długości sesji",      "median_session_length"),
        ("Sparsity",                    "sparsity"),
    ]

    all_stats = [stats_30m_raw, stats_30m_proc, stats_lastfm_raw, stats_lastfm_proc]

    for label, key in metrics:
        vals = [s[key] for s in all_stats]
        if key == "sparsity":
            fmts = [f"{v*100:.2f}%" for v in vals]
        elif isinstance(vals[0], float):
            fmts = [f"{v:.2f}" for v in vals]
        else:
            fmts = [f"{v:,}" for v in vals]
        print(f"{label:<35} | {fmts[0]:>14} | {fmts[1]:>14} | {fmts[2]:>14} | {fmts[3]:>14}")

    print(sep)

    # ─── Write LaTeX table ───
    write_latex_table(stats_30m_raw, stats_30m_proc, stats_lastfm_raw, stats_lastfm_proc,
                      str(output_tex))


if __name__ == "__main__":
    main()
