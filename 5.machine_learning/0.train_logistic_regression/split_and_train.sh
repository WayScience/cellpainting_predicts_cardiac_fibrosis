#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the machine learning conda environment
conda activate machine_learning_cfret

# convert all notebooks to python files into the nbconverted folder
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run python nbconverted to split data and train models
python nbconverted/0.split_data.py
python nbconverted/1.train_models.py
python nbconverted/2.train_supp_models.py

