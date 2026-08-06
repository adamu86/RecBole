#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt do zliczania unikalnych utworów (item_id), sesji, użytkowników oraz interakcji 
w plikach .inter zbiorów 30Music (w katalogach dataset/ oraz dataset_processed/).

Oblicza zarówno statystyki dla poszczególnych folderów, jak i PODSUMOWANIE ŁĄCZNE
(unia/suma ze wszystkich zbiorów danych).
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_inter_file_full(inter_path):
    """
    Parsuje plik .inter i zwraca zbiory unikalnych utworów, sesji, użytkowników
    oraz łączną liczbę interakcji.
    """
    unique_items = set()
    unique_sessions = set()
    unique_users = set()
    total_interactions = 0

    with open(inter_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
        sep = "\t" if "\t" in header_line else " "
        header = header_line.split(sep)

        item_idx = -1
        session_idx = -1
        user_idx = -1

        for idx, col in enumerate(header):
            col_clean = col.split(":")[0]
            if col_clean == "item_id":
                item_idx = idx
            elif col_clean == "session_id":
                session_idx = idx
            elif col_clean == "user_id":
                user_idx = idx

        if item_idx == -1:
            print(f"[OSTRZEŻENIE] Brak kolumny item_id w pliku {inter_path}")
            return None

        for line in tqdm(f, desc=f"Parsowanie {os.path.basename(inter_path)}", leave=False):
            parts = line.strip().split(sep)
            if len(parts) > item_idx:
                unique_items.add(parts[item_idx])
                total_interactions += 1

            if session_idx != -1 and len(parts) > session_idx:
                unique_sessions.add(parts[session_idx])

            if user_idx != -1 and len(parts) > user_idx:
                unique_users.add(parts[user_idx])

    return {
        "file": os.path.basename(inter_path),
        "items_set": unique_items,
        "sessions_set": unique_sessions,
        "users_set": unique_users,
        "total_interactions": total_interactions,
    }


def analyze_directory(dataset_dir):
    """
    Analizuje wszystkie pliki .inter w podanym katalogu.
    Zwraca główne statystyki do agregacji globalnej.
    """
    dir_name = os.path.basename(dataset_dir)
    all_files = os.listdir(dataset_dir)

    # 1. Główny plik .inter (np. 30music...inter, wykluczając split train/valid/test)
    main_inter = [
        f for f in all_files 
        if f.endswith('.inter') and not f.endswith(('.train.inter', '.valid.inter', '.test.inter'))
    ]

    # 2. Pliki podziału train/valid/test
    split_inters = [
        f for f in all_files 
        if f.endswith(('.train.inter', '.valid.inter', '.test.inter'))
    ]

    print("\n" + "=" * 90)
    print(f" KATALOG ZBIORU DANYCH: {dir_name}")
    print("=" * 90)

    dir_main_stats = None

    if main_inter:
        for f_name in main_inter:
            f_path = os.path.join(dataset_dir, f_name)
            stats = parse_inter_file_full(f_path)
            if stats:
                dir_main_stats = stats
                n_items = len(stats['items_set'])
                n_sessions = len(stats['sessions_set'])
                n_users = len(stats['users_set'])
                n_inter = stats['total_interactions']

                print(f"--- Główny plik interakcji (.inter): {stats['file']} ---")
                print(f"  • Liczba unikalnych utworów (item_id):  {n_items:>12,}")
                print(f"  • Liczba unikalnych sesji (session_id): {n_sessions:>12,}")
                print(f"  • Liczba unikalnych użytkowników (user):{n_users:>12,}")
                print(f"  • Łączna liczba interakcji (odsłuchań): {n_inter:>12,}")
                if n_sessions > 0:
                    print(f"  • Średnia długość sesji:                {n_inter / n_sessions:>12.2f}")

    if split_inters:
        print("\n--- Pliki podziału (train/valid/test) ---")
        for f_name in sorted(split_inters):
            f_path = os.path.join(dataset_dir, f_name)
            stats = parse_inter_file_full(f_path)
            if stats:
                n_items = len(stats['items_set'])
                n_inter = stats['total_interactions']
                print(f"  [{stats['file']}]")
                print(f"    - Unikalne utwory: {n_items:>10,} | Interakcje/Próbki: {n_inter:>10,}")

    # Sprawdzenie pliku .item (jeśli istnieje)
    item_files = [f for f in all_files if f.endswith('.item')]
    if item_files:
        item_path = os.path.join(dataset_dir, item_files[0])
        unique_in_item = 0
        with open(item_path, "r", encoding="utf-8") as f:
            f.readline()  # header
            unique_in_item = sum(1 for _ in f)
        print(f"\n--- Plik metadanych przedmiotów (.item): {item_files[0]} ---")
        print(f"  • Liczba wpisów w pliku .item:           {unique_in_item:>12,}")

    print("=" * 90)
    return dir_main_stats


def main():
    parser = argparse.ArgumentParser(
        description="Zliczanie unikalnych utworów z plików .inter zbiorów 30Music (indywidualnie i łącznie)."
    )
    parser.add_argument(
        "--dir", "-d", type=str, default=None,
        help="Ścieżka do konkretnego katalogu zbioru (np. dataset/30music__...). Domyślnie przeszukuje cały katalog dataset/."
    )

    args = parser.parse_args()

    global_items = set()
    global_sessions = set()
    global_users = set()
    global_interactions = 0
    analyzed_dirs_count = 0

    if args.dir:
        if os.path.exists(args.dir):
            stats = analyze_directory(args.dir)
            if stats:
                global_items.update(stats['items_set'])
                global_sessions.update(stats['sessions_set'])
                global_users.update(stats['users_set'])
                global_interactions += stats['total_interactions']
                analyzed_dirs_count = 1
        else:
            print(f"[BŁĄD] Katalog nie istnieje: {args.dir}")
            sys.exit(1)
    else:
        dataset_base = "dataset"
        if not os.path.exists(dataset_base):
            print(f"[BŁĄD] Katalog {dataset_base} nie istnieje.")
            sys.exit(1)

        subdirs = [
            os.path.join(dataset_base, d) for d in os.listdir(dataset_base)
            if os.path.isdir(os.path.join(dataset_base, d)) and d.startswith("30music")
        ]

        if not subdirs:
            subdirs = [
                os.path.join(dataset_base, d) for d in os.listdir(dataset_base)
                if os.path.isdir(os.path.join(dataset_base, d))
            ]

        for d in sorted(subdirs):
            stats = analyze_directory(d)
            if stats:
                global_items.update(stats['items_set'])
                global_sessions.update(stats['sessions_set'])
                global_users.update(stats['users_set'])
                global_interactions += stats['total_interactions']
                analyzed_dirs_count += 1

    print("\n" + "=" * 90)
    print(" PODSUMOWANIE ŁĄCZNE (UNIA I SUMA ZE WSZYSTKICH PRZEANALIZOWANYCH ZBIORÓW)")
    print("=" * 90)
    print(f"  • Przeanalizowane katalogi:                 {analyzed_dirs_count:>12,}")
    print(f"  • ŁĄCZNIE unikalne utwory (item_id):        {len(global_items):>12,}")
    print(f"  • ŁĄCZNIE unikalne sesje (session_id):      {len(global_sessions):>12,}")
    print(f"  • ŁĄCZNIE unikalni użytkownicy (user_id):   {len(global_users):>12,}")
    print(f"  • ŁĄCZNIE wszystkie interakcje (odsłuchania):{global_interactions:>12,}")
    if len(global_sessions) > 0:
        print(f"  • Średnia długość sesji (globalnie):          {global_interactions / len(global_sessions):>12.2f}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
