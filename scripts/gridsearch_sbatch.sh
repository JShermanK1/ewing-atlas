#! /bin/bash

cd ~/projects/ewing_atlas
eval "$(conda shell.bash hook)"
conda activate ewing_atlas
python scripts/umap_gridsearch.py
