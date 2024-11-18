#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the machine learning conda environment
conda activate machine_learning_cfret

# Go into the first module for training models
cd 0.train_logisitic_regression

# convert all notebooks to python files into the scripts folder
jupyter nbconvert --to script --output-dir=scripts/ *.ipynb

# run python scripts to split data and train models
python scripts/0.split_data.py
python scripts/1.train_models.py

