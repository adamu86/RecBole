import json
import re
import os
import sys
from typing import Counter
from tqdm import tqdm


# entities
albums = "entites/albums.idomaar"
persons = "entites/persons.idomaar"
playlist = "entites/playlist.idomaar"
tags = "entites/tags.idomaar"
tracks = "entites/tracks.idomaar"
users = "entites/users.idomaar"

# relations
sessions = "relations/sessions.idomaar"
love = "relations/love.idomaar"
events = "relations/events.idomaar"

# output directory
output_dir = "processed/"

# sessions process section
min_playcount = 5
min_tracks = 2

with open(sessions, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

def get_ts(line):
    try:
        return int(line[len("event.session\t"):].split("\t")[1])
    except (IndexError, ValueError):
        return None

tmp_file = sessions + ".tmp"

with open(sessions, "r", encoding="utf-8") as f:
    max_ts = max(
        (ts for line in tqdm(f, total=total_lines, desc="Finding max_ts")
         if (ts := get_ts(line)) is not None),
        default=0
    )

days = max_ts - 24 * 3600 * 90

raw_lines = []
with open(sessions, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total_lines, desc=f"Filtering last {days // (24 * 3600)} days"):
        ts = get_ts(line)
        if ts is not None and ts >= days:
            raw_lines.append((ts, line))

print("Sorting sessions...")
raw_lines.sort(key=lambda x: x[0])

with open(tmp_file, "w", encoding="utf-8") as f:
    for _, line in tqdm(raw_lines, desc="Writing filtered sessions"):
        f.write(line)

os.replace(tmp_file, sessions)

with open(sessions, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

with open(sessions, "r", encoding="utf-8") as fin:
    lines = []

    for line in tqdm(fin, total=total_lines, desc=f"Processing raw {sessions}"):
        line = line[len("event.session\t"):]

        try:
            parts = line.split("\t")
            session_id = int(parts[0])
            session_timestamp = int(parts[1])
            session_stats = json.loads(parts[2][:parts[2].find("} {") + 1])
            session_objects = json.loads(line[line.find("} {") + 2:].strip())
        except (json.JSONDecodeError, ValueError, IndexError):
            continue

        session_playtime = session_stats["playtime"]
        session_user_id = session_objects["subjects"][0]["id"]
        session_tracks = session_objects["objects"]
        session_tracks = [
            {
                "id": st["id"],
                "ps": st["playstart"],
                "pt": st["playtime"],
                "pr": st.get("playratio")
            }
            for st in session_objects["objects"]
            if st.get("action") == "play"
        ]

        if len(session_tracks) >= min_tracks:
            new_line = f"{session_id}\t{session_timestamp}\t{session_playtime}\t{session_user_id}\t{json.dumps(session_tracks, separators=(',', ':'))}\n"
            lines.append((session_timestamp, new_line))

print("Sorting...")
lines.sort(key=lambda x: x[0])
with open(f"{output_dir}{sessions}", "w", encoding="utf-8") as fout:
    for _, line in tqdm(lines, desc="Writing"):
        fout.write(line)

print("\n")

def count_tracks_in_sessions():
    track_counts = Counter()

    with open(f"{output_dir}{sessions}", "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Collecting track IDs"):
            parts = line.split("\t")
            tracks_data = json.loads(parts[4])
            for t in tracks_data:
                track_counts[t["id"]] += 1

    with open(f"{output_dir}track_ids.tsv", "w", encoding="utf-8") as f:
        for tid, count in sorted(track_counts.items()):
            if count >= min_playcount:
                f.write(f"{tid}\n")

    return sum(track_counts.values())

def filter_sessions_by_tracks():
    allowed_tracks = set()

    with open(f"{output_dir}track_ids.tsv", "r", encoding="utf-8") as f:
        for line in f:
            allowed_tracks.add(int(line.strip()))

    lines = []

    with open(f"{output_dir}{sessions}", "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Filtering rare tracks"):
            parts = line.split("\t")
            session_id = parts[0]
            session_timestamp = int(parts[1])
            session_playtime = parts[2]
            session_user_id = parts[3]
            tracks_data = json.loads(parts[4])

            tracks_data = [t for t in tracks_data if t["id"] in allowed_tracks]

            if len(tracks_data) >= min_tracks:
                new_line = f"{session_id}\t{parts[1]}\t{session_playtime}\t{session_user_id}\t{json.dumps(tracks_data, separators=(',', ':'))}\n"
                lines.append((session_timestamp, new_line))

    with open(f"{output_dir}{sessions}", "w", encoding="utf-8") as fout:
        for _, line in tqdm(lines, desc="Writing"):
            fout.write(line)

    return len(lines)   

# iteration = 1
# while True:
#     print(f"\nIteration {iteration}")
#     prev = count_tracks_in_sessions()
#     curr = filter_sessions_by_tracks()
#     print(f"Sessions: {curr:,}  Interactions before filtering: {prev:,}")
#     curr = count_tracks_in_sessions()
#     if prev == curr:
#         break
#     iteration += 1

count_tracks_in_sessions()
filter_sessions_by_tracks()

os.remove(f"{output_dir}track_ids.tsv")

input_path = f"{output_dir}{sessions}"
output_lines = []
null_count = 0
clipped_count = 0

with open(input_path, "r", encoding="utf-8") as fin:
    for line in tqdm(fin, desc="Processing playratio"):
        parts = line.strip().split("\t")
        tracks_data = json.loads(parts[4])
        
        for t in tracks_data:
            if t["pr"] is None:
                t["pr"] = 1.0
                null_count += 1
            if t["pr"] > 2.0:
                clipped_count += 1
            t["pr"] = min(t["pr"], 2.0)
        
        parts[4] = json.dumps(tracks_data, separators=(',', ':'))
        output_lines.append("\t".join(parts) + "\n")

with open(input_path, "w", encoding="utf-8") as fout:
    for line in tqdm(output_lines, desc="Writing"):
        fout.write(line)

print(f"Replaced nulls with 1.0: {null_count:,}")
print(f"Clipped values to 2.0: {clipped_count:,}")

input_path = "processed/relations/sessions.idomaar"
output_path = "../dataset/30music/30music.inter"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    fout.write("session_id:token\tuser_id:token\titem_id:token\ttimestamp:float\trating:float\n")
    
    for line in tqdm(fin, desc="Building .inter"):
        parts = line.strip().split("\t")
        session_id = parts[0]
        timestamp = parts[1]
        user_id = parts[3]
        tracks = json.loads(parts[4])
        
        for track in tracks:
            fout.write(f"{session_id}\t{user_id}\t{track['id']}\t{timestamp}\t{track['pr']}\n")