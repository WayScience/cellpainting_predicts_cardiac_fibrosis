#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the machine learning conda environment
conda activate machine_learning_cfret

# convert all notebooks to python files into the nbconverted folder
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run python nbconverted to evaluate the model perforamnce
python nbconverted/0.evaluate_models.py

# change to R env
conda deactivate
conda activate r_analysis_cfret

# run R script to visualize accuracy scores
Rscript nbconverted/1.vis_accuracy_scores.r
