"""Analyze track-level tag coverage in the raw 30Music dataset.

Parses tracks and tags.idomaar files to determine what percentage of tracks
have tag annotations, and what the tag distribution looks like.
"""

import json
import os
from collections import Counter
from urllib.parse import unquote_plus

TRACKS_FILE = "dataset_raw/tracks.idomaar"
TAGS_FILE = "dataset_raw/tags.idomaar"

def load_tags(tags_file):
    """Load tag_id -> tag_name mapping from tags.idomaar."""
    tag_map = {}
    with open(tags_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0] == "tag":
                tag_id = int(parts[1])
                try:
                    tag_data = json.loads(parts[3])
                    tag_name = tag_data.get("value", "")
                    tag_map[tag_id] = tag_name
                except json.JSONDecodeError:
                    pass
    return tag_map

def analyze_tracks(tracks_file, tag_map):
    """Analyze track-level tag coverage."""
    total_tracks = 0
    tracks_with_tags = 0
    tracks_without_tags = 0
    tag_counts = Counter()  # how many tracks per tag
    tags_per_track = []     # number of tags per track (for those that have tags)
    
    # Also track which item_ids have tags (for .item file coverage)
    item_ids_with_tags = set()
    item_ids_all = set()
    
    with open(tracks_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5 or parts[0] != "track":
                continue
            
            total_tracks += 1
            track_id = int(parts[1])
            item_ids_all.add(track_id)
            
            try:
                relationships = json.loads(parts[4])
                track_tags = relationships.get("tags", [])
                
                if track_tags:
                    tracks_with_tags += 1
                    item_ids_with_tags.add(track_id)
                    tag_ids = [t["id"] for t in track_tags if t.get("id") is not None]
                    tags_per_track.append(len(tag_ids))
                    
                    for tid in tag_ids:
                        tag_name = tag_map.get(tid, f"unknown_{tid}")
                        tag_counts[tag_name] += 1
                else:
                    tracks_without_tags += 1
            except (json.JSONDecodeError, KeyError):
                tracks_without_tags += 1

    # Print results
    print("=" * 70)
    print(" ANALIZA POKRYCIA TAGÓW NA POZIOMIE UTWORÓW (30Music)")
    print("=" * 70)
    
    print(f"\nŁączna liczba utworów w zbiorze:     {total_tracks:,}")
    print(f"Utwory Z tagami:                     {tracks_with_tags:,} ({tracks_with_tags/total_tracks*100:.1f}%)")
    print(f"Utwory BEZ tagów:                    {tracks_without_tags:,} ({tracks_without_tags/total_tracks*100:.1f}%)")
    
    if tags_per_track:
        import statistics
        print(f"\nŚrednia liczba tagów per utwór (wśród tych z tagami): {statistics.mean(tags_per_track):.1f}")
        print(f"Mediana:  {statistics.median(tags_per_track):.0f}")
        print(f"Max:      {max(tags_per_track)}")
    
    print(f"\nŁączna liczba unikalnych tagów:      {len(tag_counts):,}")
    
    print(f"\nTop 30 najczęstszych tagów (ile utworów je ma):")
    for tag_name, count in tag_counts.most_common(30):
        print(f"  {tag_name:30s}  {count:6,} utworów")
    
    # Check overlap with actual dataset item files
    print("\n" + "=" * 70)
    print(" POKRYCIE TAGÓW W PRZETWORZONYCH ZBIORACH DANYCH")
    print("=" * 70)
    
    dataset_base = "dataset"
    if os.path.exists(dataset_base):
        for d in sorted(os.listdir(dataset_base)):
            dir_path = os.path.join(dataset_base, d)
            if not os.path.isdir(dir_path):
                continue
            
            item_files = [f for f in os.listdir(dir_path) if f.endswith('.item')]
            if not item_files:
                continue
            
            item_path = os.path.join(dir_path, item_files[0])
            dataset_item_ids = set()
            with open(item_path, "r", encoding="utf-8") as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split("\t")
                    if parts:
                        try:
                            dataset_item_ids.add(int(parts[0]))
                        except ValueError:
                            pass
            
            overlap = dataset_item_ids & item_ids_with_tags
            total_in_dataset = len(dataset_item_ids)
            
            print(f"\n  {d}:")
            print(f"    Utworów w zbiorze:              {total_in_dataset:,}")
            print(f"    Z nich z tagami track-level:    {len(overlap):,} ({len(overlap)/total_in_dataset*100:.1f}%)")
            print(f"    Bez tagów track-level:          {total_in_dataset - len(overlap):,} ({(total_in_dataset - len(overlap))/total_in_dataset*100:.1f}%)")

if __name__ == "__main__":
    print("Wczytywanie tagów z tags.idomaar...")
    tag_map = load_tags(TAGS_FILE)
    print(f"Wczytano {len(tag_map):,} tagów.")
    
    print("Analizowanie tracks.idomaar...")
    analyze_tracks(TRACKS_FILE, tag_map)
