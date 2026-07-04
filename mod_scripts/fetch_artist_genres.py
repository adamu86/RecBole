"""
Skrypt pobierający gatunki (tagi) z Last.fm dla każdego unikalnego artysty
z pliku all_artists.tsv za pomocą pylast.

Wyjście (artist_genres.tsv):
    artysta<TAB>json_z_gatunkami

Obsługuje:
    - wznawianie po przerwaniu (pomija już pobrane wpisy)
    - rate-limiting (1s między requestami + 30s przerwa co batch)
    - exponential backoff przy błędach sieciowych
    - logowanie postępu co N artystów
"""

import os
import sys
import json
import time

import pylast
from dotenv import load_dotenv

# ── Konfiguracja ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "dataset", "all_artists.tsv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "dataset", "artist_genres.tsv")


DELAY_SECONDS = 0.25     # opóźnienie między requestami (rate-limit)
BATCH_SIZE = 100        # co ile artystów zrobić dłuższą przerwę
BATCH_PAUSE = 5        # długość przerwy między batchami (sekundy)
MAX_RETRIES = 3         # maks. liczba ponownych prób przy błędzie
LOG_EVERY = 50          # loguj postęp co N artystów

# ── Funkcje pomocnicze ────────────────────────────────────────────────────────

def load_artists(path: str) -> list[str]:
    """Wczytuje listę artystów z pliku (jedna linia = jeden artysta)."""
    artists = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                artists.append(name)
    return artists


def load_already_fetched(path: str) -> set[str]:
    """Wczytuje artystów, którzy już zostali pobrani (do wznawiania)."""
    fetched = set()
    if not os.path.isfile(path):
        return fetched
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            artist_name = line.split("\t", maxsplit=1)[0]
            fetched.add(artist_name)
    return fetched


def fetch_tags(network: pylast.LastFMNetwork, artist_name: str) -> list[dict]:
    """Pobiera wszystkie tagi artysty z Last.fm z exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            artist = network.get_artist(artist_name)
            top_tags = artist.get_top_tags()
            return [
                {"tag": t.item.name, "weight": int(t.weight)}
                for t in top_tags
            ]
        except pylast.WSError as e:
            # np. artysta nie istnieje w bazie Last.fm — nie ponawiaj
            print(f"  [WSError] {artist_name}: {e}")
            return []
        except pylast.NetworkError as e:
            wait = 2 ** attempt * 5  # 10s, 20s, 40s
            print(f"  [NetworkError] {artist_name} (próba {attempt}/{MAX_RETRIES}): {e}")
            print(f"    Czekam {wait}s przed ponowną próbą...")
            time.sleep(wait)
        except Exception as e:
            wait = 2 ** attempt * 5
            print(f"  [Error] {artist_name} (próba {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(wait)
    print(f"  [FAILED] {artist_name}: przekroczono {MAX_RETRIES} prób")
    return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Błąd: Brak zmiennej środowiskowej API_KEY w pliku .env")
        sys.exit(1)

    network = pylast.LastFMNetwork(api_key=api_key)

    # Wczytaj artystów
    artists = load_artists(INPUT_FILE)
    total = len(artists)
    print(f"Wczytano {total} artystów z {INPUT_FILE}")

    # Sprawdź, co już zostało pobrane (wznawianie)
    already_fetched = load_already_fetched(OUTPUT_FILE)
    skipped = len(already_fetched)
    if skipped:
        print(f"Pomijam {skipped} już pobranych artystów (wznawianie)")

    # Pobieraj i dopisuj wyniki
    fetched_count = 0
    error_count = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for i, artist_name in enumerate(artists, start=1):
            if artist_name in already_fetched:
                continue

            tags = fetch_tags(network, artist_name)
            tags_json = json.dumps(tags, ensure_ascii=False)
            out.write(f"{artist_name}\t{tags_json}\n")
            out.flush()

            fetched_count += 1
            if not tags:
                error_count += 1

            if fetched_count % LOG_EVERY == 0:
                print(f"  [{i}/{total}] Pobrano {fetched_count} artystów "
                      f"(błędy/puste: {error_count})")

            # Dłuższa przerwa co BATCH_SIZE artystów
            if fetched_count % BATCH_SIZE == 0:
                print(f"  Przerwa {BATCH_PAUSE}s po batchu {fetched_count}...")
                time.sleep(BATCH_PAUSE)
            else:
                time.sleep(DELAY_SECONDS)

    print(f"\nZakończono! Pobrano tagów dla {fetched_count} artystów "
          f"(błędy/puste: {error_count})")
    print(f"Wynik zapisano do: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
