import sys
import os

def count_unique_artists(file_path: str) -> int:
    """Zlicza unikalnych artystów w zadanym pliku tracks.tsv."""
    artists = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Format: item_id\tArtist/_/Track
                parts = line.split("\t", maxsplit=1)
                if len(parts) < 2:
                    continue
                
                artist_track = parts[1]
                # Artysta to wszystko przed pierwszym "/_/"
                if "/_/" in artist_track:
                    artist = artist_track.split("/_/", maxsplit=1)[0].strip()
                else:
                    artist = artist_track.strip()
                
                if artist:
                    artists.add(artist)
        
        return len(artists)
    except Exception as e:
        print(f"Wystąpił błąd podczas odczytu pliku: {e}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python count_artists.py <ścieżka_do_tracks.tsv>")
        sys.exit(1)
        
    target_file = sys.argv[1]
    
    if not os.path.isfile(target_file):
        print(f"Błąd: Plik '{target_file}' nie istnieje.")
        sys.exit(1)
        
    count = count_unique_artists(target_file)
    if count >= 0:
        print(f"Liczba unikalnych artystów w pliku '{target_file}': {count}")
