import os
import json
import time
import re
import glob
import requests
import musicbrainzngs
from pylast import LastFMNetwork, WSError, NetworkError
from dotenv import load_dotenv

# MusicBrainz config
APP_NAME = "MasterThesisMusicRecommender"
APP_VERSION = "1.0"
CONTACT = "82857@student.pb.edu.pl"
USER_AGENT = f"{APP_NAME}/{APP_VERSION} ({CONTACT})"
musicbrainzngs.set_useragent(APP_NAME, APP_VERSION, CONTACT)

# LastFM config
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

def get_artist_tags_musicbrainz(artist_name: str) -> list[str]:
    """Search MusicBrainz for artist and fetch their genres in 1 single HTTP request."""
    try:
        response = requests.get(
            "https://musicbrainz.org/ws/2/artist",
            params={"query": f'artist:"{artist_name}"', "inc": "genres", "fmt": "json", "limit": 5},
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        if response.status_code == 429:
            time.sleep(2.0)
            response = requests.get(
                "https://musicbrainz.org/ws/2/artist",
                params={"query": f'artist:"{artist_name}"', "inc": "genres", "fmt": "json", "limit": 5},
                headers={"User-Agent": USER_AGENT},
                timeout=15,
            )

        if response.status_code != 200:
            return []

        data = response.json()
        artists = data.get("artists", [])
        if not artists:
            return []

        exact = [a for a in artists if a.get("name", "").casefold() == artist_name.casefold()]
        best = exact[0] if exact else artists[0]
        genres = sorted(best.get("genres", []), key=lambda g: int(g.get("count", 0)), reverse=True)
        return [normalize_tag(g["name"]) for g in genres if g.get("name")]
    except Exception as e:
        print(f"  [MB Error]: {e}")
        return []


def get_artist_tags_lastfm(artist_name: str) -> list[dict]:
    if not NETWORK:
        print("    [Warning] No connection to Last.fm (missing API key).")
        return []
    try:
        artist = NETWORK.get_artist(artist_name)
        top_tags = artist.get_top_tags(limit=20)
        return [{"tag": normalize_tag(t.item.name), "weight": int(t.weight)} for t in top_tags]
    except WSError as e:
        print(f"    [LastFM WSError]: {e}")
    except NetworkError as e:
        print(f"    [LastFM NetworkError]: {e}")
    except Exception as e:
        print(f"    [LastFM Error]: {e}")
    return []


def fetch_artist_tags(retry_empty=False, use_mb=False):
    existing_tags = {}
    if os.path.exists(ARTIST_TAGS_FILE):
        with open(ARTIST_TAGS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip('\n').split('\t')
                if len(parts) >= 4:
                    artist = parts[1]
                    try:
                        lastfm_tags = json.loads(parts[2])
                        mb_tags = json.loads(parts[3])
                        if retry_empty and not lastfm_tags and not mb_tags:
                            continue    
                        existing_tags[artist] = (lastfm_tags, mb_tags)
                    except json.JSONDecodeError:
                        pass
                        
    artist_to_tracks = {}
    with open(ARTISTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip('\n').split('\t')
            if len(parts) >= 2:
                track_id = parts[0]
                artist = parts[1]
                if artist not in artist_to_tracks:
                    artist_to_tracks[artist] = []
                artist_to_tracks[artist].append(track_id)
                    
    all_artists = sorted(list(artist_to_tracks.keys()), key=str.lower)
                
    print(f"Found tags for {len(existing_tags)}/{len(all_artists)} artists.")
    
    with open(ARTIST_TAGS_FILE, 'w', encoding='utf-8') as f_out:
        for i, artist_name in enumerate(all_artists): 
            if artist_name in existing_tags:
                lastfm_tags, mb_tags = existing_tags[artist_name]
                lastfm_json = json.dumps(lastfm_tags, ensure_ascii=False)
                mb_json = json.dumps(mb_tags, ensure_ascii=False)
                for track_id in artist_to_tracks[artist_name]:
                    f_out.write(f"{track_id}\t{artist_name}\t{lastfm_json}\t{mb_json}\n")
                continue
            
            print(f"[{i+1}/{len(all_artists)}] Searching: {artist_name}")      
            lastfm_tags = []
            mb_tags = []
            
            time.sleep(0.21)
            lastfm_fetched = get_artist_tags_lastfm(artist_name) 
            if lastfm_fetched:
                print(f"  -> Found {len(lastfm_fetched)} tags in Last.fm.")
                lastfm_tags = lastfm_fetched
            else:
                print(f"  -> No tags in Last.fm.")
                
            if use_mb:
                time.sleep(1.05)
                try:
                    mb_fetched = get_artist_tags_musicbrainz(artist_name)
                    if mb_fetched:
                        lastfm_tag_names = {t["tag"] for t in lastfm_tags}
                        added = 0
                        for t in mb_fetched:
                            if t not in lastfm_tag_names:
                                mb_tags.append(t)
                                added += 1
                        print(f"  -> Added {added} supplemental tags from MusicBrainz.")
                    else:
                        print(f"  -> No supplemental tags in MusicBrainz.")
                except Exception as e:
                    print(f"  [MB Error] fetching tags: {e}")
            
            existing_tags[artist_name] = (lastfm_tags, mb_tags)
            lastfm_json = json.dumps(lastfm_tags, ensure_ascii=False)
            mb_json = json.dumps(mb_tags, ensure_ascii=False)
            for track_id in artist_to_tracks[artist_name]:
                f_out.write(f"{track_id}\t{artist_name}\t{lastfm_json}\t{mb_json}\n")
            f_out.flush()
            
    print(f"\nFinished writing to file: {ARTIST_TAGS_FILE}")

def merge_artists():
    def extract_artists_from_tracks(tracks_path: str) -> dict[str, str]:
        track_artist_map = {}
        with open(tracks_path, encoding="utf-8") as f:
            for line in f:
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

    pattern = os.path.join(DATASET_DIR, "30music__days*", "tracks.tsv")
    tracks_files = sorted(glob.glob(pattern))

    if not tracks_files:
        print(f"Couldn't find tracks.tsv in {DATASET_DIR}")
        return

    all_tracks: dict[str, str] = {}
    for tracks_path in tracks_files:
        folder_name = os.path.basename(os.path.dirname(tracks_path))
        track_artist_map = extract_artists_from_tracks(tracks_path)
        print(f"  {folder_name}: {len(track_artist_map)} tracks")
        all_tracks.update(track_artist_map)

    print(f"\nUnique tracks: {len(all_tracks)}")

    try:
        sorted_tracks = sorted(all_tracks.items(), key=lambda x: int(x[0]))
    except ValueError:
        sorted_tracks = sorted(all_tracks.items(), key=lambda x: x[0])
        
    with open(ARTISTS_FILE, "w", encoding="utf-8") as f:
        for track_id, artist in sorted_tracks:
            f.write(f"{track_id}\t{artist}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Downloads tags for artists.")
    parser.add_argument('--retry-empty', action='store_true', help="Retry fetching tags for artists that have no tags")
    parser.add_argument('--use-mb', action='store_true', help="Add suplemental MusicBrainz artists tags")
    args = parser.parse_args()

    if not os.path.exists(ARTISTS_FILE):
        merge_artists()
    fetch_artist_tags(retry_empty=args.retry_empty, use_mb=args.use_mb)
