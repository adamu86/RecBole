import json
import pandas as pd
import os
import hashlib

class LastFMMapper:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def _hash_to_int(self, s):
        h = hashlib.md5(s.encode('utf-8')).hexdigest()[:8]
        return int(h, 16)

    def user_to_token(self, user):
        return self._hash_to_int(user)

    def item_to_token(self, name, artist):
        key = f"{name} - {artist}"
        return self._hash_to_int(key)

    def token_to_item(self, token, reverse_map):
        return reverse_map.get(token)

    def process_dataset(self, take_half=True):
        rows = []
        reverse_map_user = {}
        reverse_map_item = {}

        items = list(self.data.items())
        if take_half:
            items = items[:len(items)//2]

        for user, tracks in items:
            user_token = self.user_to_token(user)
            reverse_map_user[user_token] = user

            for entry in tracks:
                if not entry:
                    continue
                name, artist, ts = entry.split("╎")
                item_token = self.item_to_token(name, artist)
                reverse_map_item[item_token] = f"{name} - {artist}"
                rows.append([user_token, item_token, int(ts)])

        df = pd.DataFrame(
            rows,
            columns=['user_id:token', 'item_id:token', 'timestamp:float']
        )
        os.makedirs('dataset/lastfm', exist_ok=True)
        df.to_csv('dataset/lastfm/lastfm.inter', index=False, sep="\t")
        return df, reverse_map_user, reverse_map_item
    

if __name__ == "__main__":
    mapper = LastFMMapper('lastfm_data/recent_tracks.json')
    df, rev_user, rev_item = mapper.process_dataset()