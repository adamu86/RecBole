"""Script to extract track_id -> artist mapping from all 5 dataset splits
(union of all tracks present in dataset/30music__days*/tracks.tsv)
and save the result to dataset/artists.tsv.
"""

import os
import glob


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    output_file = os.path.join(dataset_dir, "artists.tsv")

    pattern = os.path.join(dataset_dir, "30music__days*", "tracks.tsv")
    tracks_files = sorted(glob.glob(pattern))

    if not tracks_files:
        print(f"Błąd: Nie znaleziono plików tracks.tsv w {dataset_dir}/30music__days*/")
        return

    print(f"Znaleziono {len(tracks_files)} plików splitów:")
    all_tracks: dict[int, str] = {}

    for tracks_path in tracks_files:
        folder_name = os.path.basename(os.path.dirname(tracks_path))
        count_in_split = 0
        with open(tracks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue
                try:
                    track_id = int(parts[0])
                except ValueError:
                    continue
                
                artist_title = parts[1]
                artist = artist_title.split("/_/")[0].strip() if "/_/" in artist_title else artist_title.strip()
                if artist:
                    all_tracks[track_id] = artist
                    count_in_split += 1
        print(f"  • {folder_name}: {count_in_split:,} utworów")

    print(f"\nŁączna liczba unikalnych utworów ze wszystkich {len(tracks_files)} splitów (suma): {len(all_tracks):,}")

    os.makedirs(dataset_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        for track_id, artist in sorted(all_tracks.items()):
            fout.write(f"{track_id}\t{artist}\n")

    print(f"Zapisano pomyślnie do: {output_file}")


if __name__ == "__main__":
    main()
