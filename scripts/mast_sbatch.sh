#! /bin/bash

cd ~/projects/ewing_atlas
eval "$(conda shell.bash hook)"
conda activate mast
Rscript scripts/mast.R
