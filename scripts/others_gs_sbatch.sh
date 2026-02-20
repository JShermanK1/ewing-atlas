#! /bin/bash

cd ~/projects/ewing_atlas
eval "$(conda shell.bash hook)"
conda activate ewing_atlas
python scripts/umap_w_others_gs.py
