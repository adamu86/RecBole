#!/bin/bash

python process_30music.py

python process_lastfm1k.py

python run_recbole.py

python run_reranking.py

python run_results.py