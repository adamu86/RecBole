#!/usr/bin/env python3
"""
Oblicza mediannę i średnią długość sesji (liczba elementów) dla obu zbiorów danych.
Obsługuje format TSV z kolumną JSON zawierającą listę elementów sesji.
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral", "serif"],
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

DATASETS = {
    "LastFM-1K": "dataset_raw/lastfm_sessions_filtered.tsv",
    "30Music":   "dataset_raw/sessions_filtered.tsv",
}


def count_items_in_json(json_str: str) -> int:
    """Parsuje kolumnę JSON i zwraca liczbę elementów sesji."""
    try:
        items = json.loads(json_str)
        return len(items)
    except (json.JSONDecodeError, TypeError):
        return 0


def compute_stats(filepath: str) -> dict | None:
    path = Path(filepath)

    lengths = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            # Kolumna z JSON to ostatnia kolumna
            if len(parts) < 1:
                continue
            json_col = parts[-1]
            n = count_items_in_json(json_col)
            if n > 0:
                lengths.append(n)

    if not lengths:
        print(f"  [BRAK DANYCH] Nie znaleziono żadnych sesji w {filepath}", file=sys.stderr)
        return None

    return {
        "count":  len(lengths),
        "mean":   statistics.mean(lengths),
        "median": statistics.median(lengths),
        "min":    min(lengths),
        "max":    max(lengths),
        "stdev":  statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
    }


def collect_item_counts(filepath: str) -> Counter:
    """Zlicza wystąpienia każdego elementu (id) we wszystkich sesjach."""
    counter: Counter = Counter()
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            try:
                items = json.loads(parts[-1])
                for item in items:
                    counter[item["id"]] += 1
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    return counter


def plot_longtail(name: str, filepath: str, out_path: str | None = None) -> None:
    """
    Generuje wykres long-tail popularności elementów dla danego zbioru.

    Parametry
    ---------
    name      : etykieta zbioru (tytuł wykresu)
    filepath  : ścieżka do pliku TSV z sesjami
    out_path  : opcjonalna ścieżka zapisu PNG; jeśli None – wyświetla interaktywnie
    """
    print(f"[{name}] Zliczanie popularności elementów...")
    counter = collect_item_counts(filepath)
    
    # Posortowane malejąco liczby wystąpień
    counts = sorted(counter.values(), reverse=True)
    ranks  = range(1, len(counts) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(ranks, counts, alpha=0.15, color="#2a9d8f")
    ax.plot(ranks, counts, color="#2a9d8f", linewidth=1.2)

    ax.set_yscale("log")
    ax.set_xlabel("Ranking elementu")
    ax.set_ylabel("Liczba wystąpień (log)")
    ax.set_title(f"Long-tail popularności elementów – {name}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path)
        print(f"  Zapisano: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_session_lengths(
    name: str,
    filepath: str,
    out_path: str | None = None,
    max_length: int = 50,
    color: str = "#2a9d8f",
) -> None:
    """
    Histogram rozkładu długości sesji (liczba elementów).

    Parametry
    ---------
    name       : etykieta zbioru (tytuł wykresu)
    filepath   : ścieżka do pliku TSV z sesjami
    out_path   : opcjonalna ścieżka zapisu PNG; jeśli None – wyświetla interaktywnie
    max_length : sesje dłuższe niż ta wartość są wrzucane do ostatniego koszyka "max+"
    color      : kolor słupków
    """
    print(f"[{name}] Zbieranie długości sesji...")
    lengths: list[int] = []
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            try:
                items = json.loads(parts[-1])
                n = len(items)
                if n > 0:
                    lengths.append(min(n, max_length))
            except (json.JSONDecodeError, TypeError):
                continue

    if not lengths:
        print(f"  [BRAK DANYCH] {filepath}")
        return

    import numpy as np
    bins = np.arange(1, max_length + 2) - 0.5          # wyśrodkowane słupki
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=bins, color=color, alpha=0.85, edgecolor="none", rwidth=0.85)

    ax.set_yscale("log")
    ax.set_xlabel("Długość sesji (liczba elementów)", fontweight="bold", fontsize=11, labelpad=6)
    ax.set_ylabel("Liczba sesji (log)", fontweight="bold", fontsize=11, labelpad=6)
    ax.set_title(f"Rozkład długości sesji – {name}")

    # Etykiety osi X co 5, ostatnia to "max+"
    ticks = [1] + list(range(5, max_length + 1, 5))
    labels = ["1"] + [str(t) for t in range(5, max_length, 5)] + [f"{max_length}+"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold", rotation=45, ha="right")
    ax.set_xlim(0.4, max_length + 0.6)

    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_axisbelow(True)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path)
        print(f"  Zapisano: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    # print("=" * 55)
    # print(f"{'Statystyki dlugosci sesji (liczba elementow)':^55}")
    # print("=" * 55)

    # for name, path in DATASETS.items():
    #     print(f"\n{name}  ({path})")
    #     stats = compute_stats(path)
    #     if stats is None:
    #         continue
    #     print(f"  Liczba sesji : {stats['count']:>10,}")
    #     print(f"  Srednia      : {stats['mean']:>10.2f}")
    #     print(f"  Mediana      : {stats['median']:>10.2f}")
    #     print(f"  Min          : {stats['min']:>10}")
    #     print(f"  Max          : {stats['max']:>10}")
    #     print(f"  Odch. std.   : {stats['stdev']:>10.2f}")

    # print("\n" + "=" * 55)

    # --- Long-tail plots ---
    for name, path in DATASETS.items():
        if Path(path).exists():
            out_lt = path.replace(".tsv", "_longtail.png")
            plot_longtail(name, path, out_path=out_lt)

            out_sl = path.replace(".tsv", "_session_lengths.png")
            plot_session_lengths(name, path, out_path=out_sl)


if __name__ == "__main__":
    main()
