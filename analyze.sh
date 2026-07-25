#!/bin/bash

# Generowanie wykresów zbiorczych:
# python analyze_sessions.py --file dataset_raw/sessions.tsv --out analysis_plots_processed

# Dedykowana analiza spójności artystów i gatunków w sessions.tsv:
python analyze_sessions.py --file dataset_raw/sessions.tsv --check-artists
python analyze_sessions.py --file dataset_raw/sessions.tsv --check-genres
