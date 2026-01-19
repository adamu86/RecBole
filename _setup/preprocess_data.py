import json
import os
import pandas as pd

with open('lastfm_data/recent_tracks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

filtered_data = {user: tracks for user, tracks in data.items() if len(tracks) >= 10}

user_map = {user: i+1 for i, user in enumerate(filtered_data.keys())}
item_map = {}
item_counter = 1

rows = []
for user, tracks in list(filtered_data.items())[:1000]:
    for entry in tracks:
        if not entry:
            continue

        name, artist, ts = entry.split("╎")

        item_key = f"{name} - {artist}"

        if item_key not in item_map:
            item_map[item_key] = item_counter
            item_counter += 1

        rows.append([user_map[user], item_map[item_key], int(ts)])

df = pd.DataFrame(
    rows, 
    columns=['user_id:token', 'item_id:token', 'timestamp:float']
)

os.makedirs(
    name='dataset/lastfm', 
    exist_ok=True
)

df.to_csv(
    path_or_buf='dataset/lastfm/lastfm.inter', 
    index=False,
    sep="\t"
)