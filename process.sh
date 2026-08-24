#!/bin/bash

# sed -i 's/\r$//' process.sh

# params=(
#     "165 145"
#     "145 125"
#     "125 105"
#     "105 85"
#     "85 65"
# )

# for p in "${params[@]}"; do
#     read from to <<< "$p"
#     python3 process_30music.py --days_from_max $from --days_to_max $to
# done

# python process_30music.py

python process_lastfm1k.py

# python run_recbole.py
