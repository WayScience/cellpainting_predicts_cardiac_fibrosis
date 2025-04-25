#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the machine learning conda environment
conda activate machine_learning_cfret

# convert all notebooks to python files into the nbconverted folder
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run python nbconverted to extract apply model to hold out plates 1, 2, and 3
python nbconverted/0.plate3_generalizability.py
python nbconverted/4.dose_generalizability.py
python nbconverted/7.drug_x_probabilities_plate3.py

# change to R env
conda deactivate
conda activate r_analysis_cfret

# run R script to visualize generalizability
Rscript nbconverted/1.vis_plate3_generalizability.r
Rscript nbconverted/2.plate3_prob_UMAP.r
Rscript nbconverted/3.plate3_actin_feature_UMAP.r
Rscript nbconverted/5.vis_dose_generalizability.r
