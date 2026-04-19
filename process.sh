# !/bin/bash

params=(
    "120 119"
    "118 116"
    "115 112"
)

for p in "${params[@]}"; do
    read from to <<< "$p"
    python process.py --days_from_max $from --days_to_max $to
done

python run_recbole.py