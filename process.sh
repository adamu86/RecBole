#!/bin/bash

params=(
    # overlapping, gap=100
    "365 265"
    "315 215"
    "265 165"
    "215 115"
    "165 65"
    # non-overlapping, gap=60
    "365 305"
    "305 245"
    "245 185"
    "185 125"
    "125 65"
)

for p in "${params[@]}"; do
    read from to <<< "$p"
    python process.py --days_from_max $from --days_to_max $to
done

python run_recbole.py