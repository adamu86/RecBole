import json
import pandas as pd

# --- Przetworzone datasety (TSV z kolumnami timestamp + user_id) ---
FILES = {
    "30Music": "dataset_raw/sessions_filtered.tsv",
    "LastFM-1k": "dataset_raw/lastfm_sessions_filtered.tsv",
}

for name, path in FILES.items():
    print(f"\n{'='*40}\n{name}: {path}\n{'='*40}")
    
    # Wczytanie tylko kolumny timestamp (1) i user_id (2)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[1, 2],
        names=["timestamp", "user_id"],
    )
    
    # Sortowanie chronologiczne
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    n = len(df)
    
    print(f"Liczba wszystkich sesji:      {n:,}")
    print(f"Liczba unikalnych userów:     {df['user_id'].nunique():,}")


# --- Surowe dane: sessions.idomaar (30Music) ---
# Format kolumn: event_type, session_id, timestamp, props_json, subjects_objects_json
# user_id wyciągany z pola subjects: [{"type":"user","id":...}]
print(f"\n{'='*40}\n30Music RAW: dataset_raw/sessions.idomaar\n{'='*40}")
idomaar_path = "dataset_raw/sessions.idomaar"
idomaar_users = set()
with open(idomaar_path, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        try:
            # parts[3] zawiera dwie JSONy: {"numtracks":...} {"subjects":...}
            # subjects JSON zaczyna się po pierwszym "} {"
            sep = parts[3].find("} {")
            if sep == -1:
                continue
            subjects_str = parts[3][sep + 2:]
            payload = json.loads(subjects_str)
            for subj in payload.get("subjects", []):
                if subj.get("type") == "user":
                    idomaar_users.add(subj["id"])
        except (json.JSONDecodeError, IndexError):
            continue
print(f"Liczba unikalnych userów (surowe): {len(idomaar_users):,}")


# --- Surowe dane: userid-timestamp-artid-artname-traid-traname.tsv (LastFM-1k) ---
# Format: user_id \t timestamp \t artist_id \t artist_name \t track_id \t track_name
# user_id to pierwsza kolumna (np. "user_001000")
print(f"\n{'='*40}\nLastFM-1k RAW: dataset_raw/userid-timestamp-artid-artname-traid-traname.tsv\n{'='*40}")
lastfm_path = "dataset_raw/userid-timestamp-artid-artname-traid-traname.tsv"
lastfm_df = pd.read_csv(
    lastfm_path,
    sep="\t",
    header=None,
    usecols=[0],
    names=["user_id"],
)
print(f"Liczba wszystkich zdarzeń:         {len(lastfm_df):,}")
print(f"Liczba unikalnych userów (surowe): {lastfm_df['user_id'].nunique():,}")
