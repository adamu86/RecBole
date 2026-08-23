import os
import json
import time
import re
import glob
from pylast import LastFMNetwork, WSError, NetworkError
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
NETWORK = LastFMNetwork(api_key=api_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ARTISTS_FILE = os.path.join(DATASET_DIR, "artists.tsv")
ARTIST_TAGS_FILE = os.path.join(DATASET_DIR, "artists_tags.tsv")

def normalize_tag(tag: str) -> str:
    tag = tag.casefold().strip()
    tag = re.sub(r"[\s_]+", "-", tag)
    tag = re.sub(r"[^a-z0-9\-]", "", tag)
    tag = re.sub(r"-+", "-", tag).strip("-")
    return tag

def get_artist_tags_lastfm(artist_name: str) -> list[dict]:
    try:
        artist = NETWORK.get_artist(artist_name)
        top_tags = artist.get_top_tags(limit=20)

        return [{"tag": normalize_tag(tag.item.name), "weight": int(tag.weight)} for tag in top_tags]
    except WSError as e:
        print(f"Error: {e}")
    except NetworkError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    return []

def fetch_artist_tags():
    existing_tags = {}
    if os.path.exists(ARTIST_TAGS_FILE):
        with open(ARTIST_TAGS_FILE, 'r', encoding='utf-8') as file_in:
            for line in file_in:
                parts = line.strip('\n').split('\t')

                if len(parts) >= 3:
                    artist = parts[1]
                    try:
                        lastfm_tags = json.loads(parts[2])
                        existing_tags[artist] = lastfm_tags
                    except json.JSONDecodeError:
                        pass
                        
    artist_to_tracks = {}
    with open(ARTISTS_FILE, 'r', encoding='utf-8') as file_in:
        for line in file_in:
            parts = line.strip('\n').split('\t')

            if len(parts) >= 2:
                track_id = parts[0]
                artist = parts[1]

                if artist not in artist_to_tracks:
                    artist_to_tracks[artist] = []

                artist_to_tracks[artist].append(track_id)
                    
    all_artists = sorted(list(artist_to_tracks.keys()), key=str.lower)
                
    print(f"Found tags for {len(existing_tags)}/{len(all_artists)} artists.")
    
    with open(ARTIST_TAGS_FILE, 'w', encoding='utf-8') as file_out:
        for i, artist_name in enumerate(all_artists): 
            if artist_name in existing_tags:
                lastfm_tags = existing_tags[artist_name]
                lastfm_json = json.dumps(lastfm_tags, ensure_ascii=False)

                for track_id in artist_to_tracks[artist_name]:
                    file_out.write(f"{track_id}\t{artist_name}\t{lastfm_json}\n")

                continue
            
            print(f"[{i+1}/{len(all_artists)}] Searching: {artist_name}")      
            lastfm_tags = []
            
            time.sleep(0.21)
            lastfm_fetched = get_artist_tags_lastfm(artist_name) 

            if lastfm_fetched:
                print(f"Found {len(lastfm_fetched)} tags")
                lastfm_tags = lastfm_fetched
            else:
                print(f"Found no tags")
            
            existing_tags[artist_name] = lastfm_tags
            lastfm_json = json.dumps(lastfm_tags, ensure_ascii=False)

            for track_id in artist_to_tracks[artist_name]:
                file_out.write(f"{track_id}\t{artist_name}\t{lastfm_json}\n")

            file_out.flush()

def merge_artists():
    def extract_artists_from_tracks(tracks_path: str) -> dict[str, str]:
        track_artist_map = {}
        with open(tracks_path, encoding="utf-8") as file_in:
            for line in file_in:
                line = line.strip()

                if not line:
                    continue

                parts = line.split("\t", maxsplit=1)

                if len(parts) < 2:
                    continue

                track_id = parts[0]
                artist_track = parts[1]

                if "/_/" in artist_track:
                    artist = artist_track.split("/_/", maxsplit=1)[0].strip()
                else:
                    artist = artist_track.strip()

                if artist:
                    track_artist_map[track_id] = artist

        return track_artist_map

    pattern = os.path.join(DATASET_DIR, "*", "tracks.tsv")
    tracks_files = sorted(glob.glob(pattern))

    if not tracks_files:
        return

    all_tracks: dict[str, str] = {}
    for tracks_path in tracks_files:
        track_artist_map = extract_artists_from_tracks(tracks_path)
        all_tracks.update(track_artist_map)

    try:
        sorted_tracks = sorted(all_tracks.items(), key=lambda x: int(x[0]))
    except ValueError:
        sorted_tracks = sorted(all_tracks.items(), key=lambda x: x[0])
        
    with open(ARTISTS_FILE, "w", encoding="utf-8") as file_out:
        for track_id, artist in sorted_tracks:
            file_out.write(f"{track_id}\t{artist}\n")

if __name__ == '__main__':
    merge_artists()
    fetch_artist_tags()
