#!/bin/bash

python3 analyze_sessions.py --file dataset_raw/sessions.idomaar --out analysis_plots_raw
python3 analyze_sessions.py --file dataset_raw/sessions.tsv --out analysis_plots_processed