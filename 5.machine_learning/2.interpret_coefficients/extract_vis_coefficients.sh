#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the machine learning conda environment
conda activate machine_learning_cfret

# convert all notebooks to python files into the nbconverted folder
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run python nbconverted to extract final model weighted coefficients and find sc crops for top features per channel
python nbconverted/0.extract_model_coef.py
python nbconverted/2.find_sc_crops_coef.py

# change to R env
conda deactivate
conda activate r_analysis_cfret

# run R script to visualize model coefficients
Rscript nbconverted/1.visualize_model_coef.r
