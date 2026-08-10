#!/bin/bash

# sed -i 's/\r$//' process.sh

params=(
    "215 185"
    "185 155"
    "155 125"
    "125 95"
    "95 65"
)

for p in "${params[@]}"; do
    read from to <<< "$p"
    python3 process_30music.py --days_from_max $from --days_to_max $to
done

python process_lastfm.py --all_splits

python run_recbole.py
