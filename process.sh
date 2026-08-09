#!/bin/bash

params=(
    # non-overlapping, gap=60
    # "365 305"
    # "305 245"
    # "245 185"
    # "185 125"
    # "125 65"
    "215 185"
    "185 155"
    "155 125"
    "125 95"
    "95 65"
)

for p in "${params[@]}"; do
    read from to <<< "$p"
    python3 process_30music.py --days_from_max $from --days_to_max $to --min_track_playcount 5
done

python process_lastfm.py --all_splits

python run_recbole.py

# sed -i 's/\r$//' process.sh