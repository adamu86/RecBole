"""Analyse session data and generate publication-quality plots.

Supports both raw ``.idomaar`` and processed ``.tsv`` formats.  The input
format is auto-detected from the file extension, or can be forced via
the ``--format`` flag.

Usage::

    python3 analyze_sessions.py --file dataset_raw/sessions.idomaar --out analysis_plots_idomaar
    python3 analyze_sessions.py --file dataset_raw/sessions.tsv     --out analysis_plots
"""

import json
import os
import argparse
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

try:
    import orjson as fast_json
except ImportError:
    fast_json = json  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SessionStats:
    """Aggregated statistics collected during a single analysis pass."""

    session_lengths: list = field(default_factory=list)
    session_durations: list = field(default_factory=list)
    item_counts: Counter = field(default_factory=Counter)
    session_dates: list = field(default_factory=list)
    total_interactions: int = 0


# ---------------------------------------------------------------------------
# Publication-quality plot configuration
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#2274A5",       # steel blue
    "secondary": "#D64933",     # vermillion
    "accent": "#6B4C9A",        # muted purple
    "highlight": "#1B998B",     # teal
    "fill_alpha": 0.20,
    "bar_palette": "muted",
}

FIG_SIZE_STANDARD = (6.5, 4.0)
FIG_SIZE_WIDE = (7.5, 3.5)
OUTPUT_DPI = 300
OUTPUT_FORMATS = ("png", "pdf")


def _setup_plot_style():
    """Configure matplotlib and seaborn for publication-quality output."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)

    mpl.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,

        # Axes and spines
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,

        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,

        # Layout
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": OUTPUT_DPI,
    })


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_figure(fig, output_dir, basename):
    """Save *fig* as PNG and PDF to *output_dir*."""
    for fmt in OUTPUT_FORMATS:
        fig.savefig(os.path.join(output_dir, f"{basename}.{fmt}"),
                    format=fmt, dpi=OUTPUT_DPI)


def _fmt_thousands(x, _pos):
    """Tick formatter: ``12000`` → ``12,000``."""
    return f"{int(x):,}"


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _plot_session_lengths(stats, output_dir, label):
    """Histogram of session lengths (number of tracks per session)."""
    data = stats.session_lengths
    max_len = max(data)

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)
    ax.hist(data, bins=range(min(data), min(max_len + 2, 105)),
            color=COLORS["primary"], edgecolor="white", linewidth=0.4, alpha=0.85)

    ax.set_title(f"Session Length Distribution — {label}", fontweight="bold")
    ax.set_xlabel("Number of tracks per session")
    ax.set_ylabel("Number of sessions")
    ax.set_xlim(0, min(100, max_len))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_thousands))

    _save_figure(fig, output_dir, "1_session_lengths")
    plt.close(fig)


def _plot_long_tail(stats, output_dir, label):
    """Long-tail plot of item popularity (log-scale y-axis)."""
    counts = sorted(stats.item_counts.values(), reverse=True)
    ranks = range(len(counts))

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)
    ax.plot(ranks, counts, color=COLORS["secondary"], linewidth=1.5)
    ax.fill_between(ranks, counts, alpha=COLORS["fill_alpha"],
                     color=COLORS["secondary"])

    ax.set_title(f"Item Popularity Distribution (Long Tail) — {label}",
                 fontweight="bold")
    ax.set_xlabel("Item rank (most popular first)")
    ax.set_ylabel("Play count (log scale)")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_thousands))

    _save_figure(fig, output_dir, "2_item_popularity_long_tail")
    plt.close(fig)


def _plot_sessions_per_day(stats, output_dir, label):
    """Time-series of daily session counts."""
    date_counts = pd.Series(stats.session_dates).value_counts().sort_index()
    date_counts.index = pd.to_datetime(date_counts.index)

    fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
    ax.plot(date_counts.index, date_counts.values,
            marker=".", markersize=4, linestyle="-",
            color=COLORS["highlight"], linewidth=1.2)

    ax.set_title(f"Sessions per Day — {label}", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of sessions")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_thousands))
    fig.autofmt_xdate(rotation=35, ha="right")

    _save_figure(fig, output_dir, "3_sessions_per_day")
    plt.close(fig)


_MAX_DURATION_VIS = 7200  # cap visualisation at 2 hours (seconds)


def _plot_durations(stats, output_dir, label):
    """Histogram of session durations (minutes), capped at 2 hours."""
    durations_min = [d / 60 for d in stats.session_durations if d <= _MAX_DURATION_VIS]
    if not durations_min:
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)
    ax.hist(durations_min, bins=60, color=COLORS["accent"],
            edgecolor="white", linewidth=0.4, alpha=0.80)

    # Smooth KDE overlay on a twin axis
    ax2 = ax.twinx()
    sns.kdeplot(durations_min, ax=ax2, color=COLORS["accent"], linewidth=1.5)
    ax2.set_ylabel("")
    ax2.set_yticks([])
    ax2.spines["right"].set_visible(False)

    ax.set_title(f"Session Duration Distribution (≤ 2 h) — {label}",
                 fontweight="bold")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Number of sessions")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_thousands))

    _save_figure(fig, output_dir, "4_session_durations")
    plt.close(fig)


_BUCKET_BINS = [0, 24, 50, 100, 500, 1000, float("inf")]
_BUCKET_LABELS = ["< 25", "25–50", "51–100", "101–500", "501–1000", "1000+"]


def _plot_popularity_buckets(stats, output_dir, label):
    """Bar chart of items grouped by popularity buckets.

    Returns ``(bucket_counts, bucket_interactions)`` for the console summary.
    """
    counts_series = pd.Series(list(stats.item_counts.values()))
    buckets = pd.cut(counts_series, bins=_BUCKET_BINS,
                     labels=_BUCKET_LABELS, right=True)
    bucket_counts = buckets.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)
    palette = sns.color_palette(COLORS["bar_palette"], n_colors=len(_BUCKET_LABELS))
    bars = ax.bar(range(len(bucket_counts)), bucket_counts.values,
                  color=palette, edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(bucket_counts)))
    ax.set_xticklabels(_BUCKET_LABELS)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_title(f"Items by Popularity Bucket — {label}", fontweight="bold")
    ax.set_xlabel("Play count range")
    ax.set_ylabel("Number of unique items")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_thousands))

    _save_figure(fig, output_dir, "5_item_popularity_buckets")
    plt.close(fig)

    bucket_interactions = counts_series.groupby(buckets, observed=False).sum()
    return bucket_counts, bucket_interactions



def _plot_session_playtime(stats, output_dir, label):
    """Histogram of full session playtime (seconds) with log-scale x-axis.

    Unlike :func:`_plot_durations` (capped at 2 h), this shows the entire
    range so outliers and the overall shape are visible.
    """
    durations = [d for d in stats.session_durations if d > 0]
    if not durations:
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)

    # Log-spaced bins to cover the wide range
    log_min = np.log10(max(min(durations), 1))
    log_max = np.log10(max(durations))
    bins = np.logspace(log_min, log_max, num=80)

    ax.hist(durations, bins=bins, color=COLORS["highlight"],
            edgecolor="white", linewidth=0.4, alpha=0.85)

    ax.set_xscale("log")

    # Mark the MIN_SESSION_PLAYTIME threshold
    ax.axvline(30, color=COLORS["secondary"], linestyle="--", linewidth=1.2,
               label="30 s threshold")
    ax.legend(frameon=True, framealpha=0.8)

    # Ensure 30 appears as a labelled tick alongside standard log ticks
    default_ticks = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    all_ticks = sorted(set(default_ticks + [30]))
    ax.set_xticks(all_ticks)
    ax.set_xticklabels([f"{int(t):,}" for t in all_ticks])

    ax.set_title(f"Session Playtime Distribution — {label}", fontweight="bold")
    ax.set_xlabel("Playtime (seconds, log scale)")
    ax.set_ylabel("Number of sessions")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_thousands))

    _save_figure(fig, output_dir, "6_session_playtime")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot orchestrator
# ---------------------------------------------------------------------------

def _generate_all_plots(stats, output_dir, label):
    """Generate all five analysis plots; return bucket data for summary."""
    os.makedirs(output_dir, exist_ok=True)
    _setup_plot_style()

    print("Generating session length distribution...")
    _plot_session_lengths(stats, output_dir, label)

    print("Generating item popularity (long tail)...")
    _plot_long_tail(stats, output_dir, label)

    print("Generating sessions per day...")
    _plot_sessions_per_day(stats, output_dir, label)

    print("Generating session duration distribution...")
    _plot_durations(stats, output_dir, label)

    print("Generating popularity bucket statistics...")
    bucket_counts, bucket_interactions = _plot_popularity_buckets(
        stats, output_dir, label
    )

    print("Generating session playtime distribution...")
    _plot_session_playtime(stats, output_dir, label)

    print(f"\nPlots saved to: {os.path.abspath(output_dir)}")
    return bucket_counts, bucket_interactions


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(stats, label, bucket_counts, bucket_interactions):
    """Print formatted bucket statistics and dataset summary to stdout."""
    sep = "=" * 80

    print(f"\n{sep}")
    print(f"  POPULARITY BUCKET STATISTICS — {label}")
    print(sep)
    print(f"{'Bucket (plays)':<25} | {'Unique items':<15} | "
          f"{'Total plays':<15} | {'% of all plays'}")
    print("-" * 80)

    total = stats.total_interactions
    for lbl in _BUCKET_LABELS:
        n_items = bucket_counts[lbl]
        n_inter = bucket_interactions[lbl]
        pct = (n_inter / total) * 100 if total > 0 else 0
        print(f"{lbl:<25} | {n_items:<15,} | {int(n_inter):<15,} | {pct:.2f}%")

    print(sep)

    lengths = stats.session_lengths
    avg_len = sum(lengths) / len(lengths)
    median_len = pd.Series(lengths).median()

    print(f"\n  DATASET SUMMARY — {label}")
    print("-" * 50)
    print(f"  Total sessions:         {len(lengths):>12,}")
    print(f"  Total interactions:     {total:>12,}")
    print(f"  Unique items:           {len(stats.item_counts):>12,}")
    print(f"  Mean session length:    {avg_len:>12.2f} tracks")
    print(f"  Median session length:  {median_len:>12.1f} tracks")
    print("-" * 50)


# ---------------------------------------------------------------------------
# Parsers — one per input format
# ---------------------------------------------------------------------------

_IDOMAAR_PREFIX = "event.session\t"
_IDOMAAR_PREFIX_LEN = len(_IDOMAAR_PREFIX)  # 14


def _get_line_count(file_path):
    """Return the number of lines in *file_path* using ``wc -l``."""
    try:
        return int(subprocess.check_output(["wc", "-l", file_path]).split()[0])
    except Exception:
        return None


def _parse_idomaar(file_path):
    """Parse a raw ``.idomaar`` events file into :class:`SessionStats`."""
    stats = SessionStats()
    total = _get_line_count(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading raw idomaar", total=total):
            if not line.startswith(_IDOMAAR_PREFIX):
                continue

            line = line[_IDOMAAR_PREFIX_LEN:]

            try:
                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                timestamp = int(parts[1])

                obj_str = line[line.find("} {") + 2:].strip()
                session_objects = fast_json.loads(obj_str)

                if "objects" not in session_objects:
                    continue

                tracks = session_objects["objects"]
                if not tracks:
                    continue

                stats.session_lengths.append(len(tracks))

                ps_values = [t["playstart"] for t in tracks if "playstart" in t]
                if ps_values:
                    stats.session_durations.append(max(ps_values) - min(ps_values))

                for t in tracks:
                    if "id" in t:
                        stats.item_counts[t["id"]] += 1
                        stats.total_interactions += 1

                date_str = datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                ).strftime("%Y-%m-%d")
                stats.session_dates.append(date_str)

            except (json.JSONDecodeError, ValueError, KeyError, IndexError):
                continue

    return stats


def _parse_tsv(file_path):
    """Parse a processed sessions TSV into :class:`SessionStats`.

    Expected columns: ``session_id  timestamp  user_id  tracks_json``
    """
    stats = SessionStats()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading processed sessions"):
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue

            try:
                timestamp = int(parts[1])
                tracks = json.loads(parts[3])

                if not tracks:
                    continue

                stats.session_lengths.append(len(tracks))

                ps_values = [t["ps"] for t in tracks]
                stats.session_durations.append(max(ps_values) - min(ps_values))

                for t in tracks:
                    stats.item_counts[t["id"]] += 1
                    stats.total_interactions += 1

                date_str = datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                ).strftime("%Y-%m-%d")
                stats.session_dates.append(date_str)

            except (json.JSONDecodeError, ValueError, KeyError, IndexError):
                continue

    return stats


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_FORMAT_PARSERS = {
    "idomaar": (_parse_idomaar, "Raw idomaar"),
    "tsv":     (_parse_tsv,     "Processed"),
}


def _detect_format(file_path):
    """Guess the input format from the file extension.

    Returns ``"idomaar"`` or ``"tsv"``.
    """
    if file_path.endswith(".idomaar"):
        return "idomaar"
    return "tsv"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze(file_path, output_dir, fmt=None):
    """Run the full analysis pipeline.

    Parameters
    ----------
    file_path : str
        Path to the input sessions file.
    output_dir : str
        Directory where plots will be saved.
    fmt : str or None
        ``"idomaar"`` or ``"tsv"``.  Auto-detected from extension if *None*.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    if fmt is None:
        fmt = _detect_format(file_path)

    parser_fn, label = _FORMAT_PARSERS[fmt]

    print(f"Analysing {file_path} (format: {fmt}) ...\n")
    stats = parser_fn(file_path)

    if not stats.session_lengths:
        print("No valid data found.")
        return

    bucket_counts, bucket_interactions = _generate_all_plots(stats, output_dir, label)
    _print_summary(stats, label, bucket_counts, bucket_interactions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse session data (raw idomaar or processed TSV) "
                    "and generate publication-quality plots.",
    )
    parser.add_argument(
        "--file", type=str, required=True,
        help="Path to the sessions file (.idomaar or .tsv)",
    )
    parser.add_argument(
        "--out", type=str, default="analysis_plots",
        help="Output directory for plots (default: analysis_plots)",
    )
    parser.add_argument(
        "--format", type=str, choices=["idomaar", "tsv"], default=None,
        dest="fmt",
        help="Force input format (auto-detected from extension if omitted)",
    )
    args = parser.parse_args()
    analyze(args.file, args.out, args.fmt)
