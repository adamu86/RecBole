import json
import pandas as pd
import os
import hashlib


class LastFMItemMapper:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def _hash_to_int(self, s):
        h = hashlib.md5(s.encode('utf-8')).hexdigest()[:8]
        return int(h, 16)

    def item_to_token(self, name, artist):
        key = f"{name} - {artist}"
        return self._hash_to_int(key)

    def build_item_mapping(self):
        item_map = {}

        for _, tracks in self.data.items():
            for entry in tracks:
                if not entry:
                    continue

                name, artist, _ = entry.split("╎")
                token = self.item_to_token(name, artist)

                if token not in item_map:
                    item_map[token] = f"{name} - {artist}"

        df = pd.DataFrame(
            list(item_map.items()),
            columns=['item_id', 'song']
        )

        os.makedirs('dataset/lastfm', exist_ok=True)
        df.to_csv(
            'dataset/lastfm/item_mapping.csv',
            index=False,
            sep="\t"
        )

        return df


if __name__ == "__main__":
    mapper = LastFMItemMapper('lastfm_data/recent_tracks.json')
    df_items = mapper.build_item_mapping()
