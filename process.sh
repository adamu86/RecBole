#!/bin/bash

# sed -i 's/\r$//' process.sh

python process_30music.py

python process_lastfm1k.py

python run_recbole.py
