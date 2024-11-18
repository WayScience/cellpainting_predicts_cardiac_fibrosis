#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the machine learning conda environment
conda activate machine_learning_cfret

# convert all notebooks to python files into the scripts folder
jupyter nbconvert --to script --output-dir=scripts/ *.ipynb

# run python scripts to extract apply model to hold out plates 1, 2, and 3
python scripts/0.plate3_generalizability.py
python scripts/4.dose_generalizability.py

# change to R env
conda deactivate
conda activate r_analysis_cfret

# run R script to visualize generalizability
Rscript scripts/1.vis_plate3_generalizability.r
Rscript scripts/2.plate3_prob_UMAP.r
Rscript scripts/3.plate3_actin_feature_UMAP.r
Rscript scripts/5.vis_dose_generalizability.r
