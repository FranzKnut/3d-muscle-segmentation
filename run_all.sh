#!/bin/bash
# Loop over all subdirectories in the specified data path
for p in data/MAIN\ PROJECT\ 11-25/C-\ Control/*; do
    echo "Launching with data_path: $p"
    python hi_prediction.py --data_path "$p"
    python li_prediction.py --data_path "$p"
done
for p in data/MAIN\ PROJECT\ 11-25/P-\ Patient/*; do
    echo "Launching with data_path: $p"
    python hi_prediction.py --data_path "$p"
    python li_prediction.py --data_path "$p"
done