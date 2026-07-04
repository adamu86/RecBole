from pylast import LastFMNetwork
from dotenv import load_dotenv
import json
import os

load_dotenv()
NETWORK = LastFMNetwork(api_key=os.getenv("API_KEY"))

def get_artist_tags(artist, limit=20):
    return [
        {
            "tag": t.item.name, 
            "weight": int(t.weight)
        } 
        for t in artist.get_top_tags(limit=limit)
    ]

def get_similar_artists(artist, limit=20):
    return [
        t.item.name
        for t in artist.get_similar(limit=limit)
    ]

def get_info(artist_name, network=NETWORK, limit=20):
    try:
        artist = network.get_artist(artist_name)
        return {
            "tags": get_artist_tags(artist, limit),
            "similar_artists": get_similar_artists(artist, limit),
        }
    except Exception:
        return {}
    
def format(obj, indent=2, ensure_ascii=False):
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii)

artist_name = "Queen"
artist = get_info(artist_name)
print(json.dumps({artist_name: [t["tag"] for t in artist["tags"] if t["weight"] > 1]}))