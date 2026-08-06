"""Generate all analysis plots for LastFM-1K dataset (including Sessions per Day)."""

import os
from analyze_sessions import analyze

def main():
    # Use full raw sessions file covering all ~1587 days
    raw_sessions_file = "dataset_raw/lastfm_sessions.tsv"
    if not os.path.exists(raw_sessions_file):
        raw_sessions_file = "dataset_processed/lastfm_sessions.tsv"

    out_dir = "analysis_plots_lastfm"

    if not os.path.exists(raw_sessions_file):
        print(f"Błąd: Nie znaleziono pliku {raw_sessions_file}")
        return

    print(f"Rozpoczynanie generowania wykresów dla LastFM-1K (pełny zbiór) z pliku {raw_sessions_file}...")
    analyze(file_path=raw_sessions_file, output_dir=out_dir, fmt="tsv", label="LastFM-1K")
    print("\n[OK] Wykresy zostały pomyślnie wygenerowane w katalogu: " + os.path.abspath(out_dir))

if __name__ == "__main__":
    main()
