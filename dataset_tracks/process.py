import json
from urllib.parse import unquote_plus

with open('tracks.idomaar', 'r') as f_in, open('tracks.tsv', 'w') as f_out:
   for line in f_in:
        parts = line.strip().split('\t')
        track_id = parts[1]
        meta = json.loads(parts[3])
        name = unquote_plus(meta['name'])
        f_out.write(f"{track_id}\t{name}\n")