"""
Skrypt mergujący artystów z plików tracks.tsv z 5 podziałów czasowych
w folderze dataset/ do jednego pliku all_artists.tsv.

Format wejściowy tracks.tsv (bez nagłówka):
    item_id<TAB>Artist/_/Track Title

Format wyjściowy all_artists.tsv:
    artist_name  (jedna linia = jeden unikalny artysta, posortowani alfabetycznie)
"""

import os
import glob

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
OUTPUT_FILE = os.path.join(DATASET_DIR, "all_artists.tsv")


def extract_artists_from_tracks(tracks_path: str) -> set[str]:
    """Wyciąga unikalne nazwy artystów z pliku tracks.tsv."""
    artists = set()
    with open(tracks_path, encoding="utf-8") as f:
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
    return artists


def main():
    # Znajdź wszystkie podfoldery z tracks.tsv
    pattern = os.path.join(DATASET_DIR, "30music__days*", "tracks.tsv")
    tracks_files = sorted(glob.glob(pattern))

    if not tracks_files:
        print(f"Nie znaleziono plików tracks.tsv w {DATASET_DIR}")
        return

    print(f"Znaleziono {len(tracks_files)} plików tracks.tsv:")
    for f in tracks_files:
        print(f"  - {os.path.basename(os.path.dirname(f))}")

    # Zbieraj artystów ze wszystkich podziałów
    all_artists: set[str] = set()
    for tracks_path in tracks_files:
        folder_name = os.path.basename(os.path.dirname(tracks_path))
        artists = extract_artists_from_tracks(tracks_path)
        print(f"  {folder_name}: {len(artists)} unikalnych artystów")
        all_artists |= artists

    print(f"\nŁącznie unikalnych artystów ze wszystkich podziałów: {len(all_artists)}")

    # Zapisz posortowaną listę do pliku
    sorted_artists = sorted(all_artists, key=str.lower)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for artist in sorted_artists:
            f.write(artist + "\n")

    print(f"Zapisano do: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
