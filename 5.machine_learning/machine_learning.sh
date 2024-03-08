#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the main conda environment
conda activate cfret_data_env

# convert all notebooks to python files into the scripts folder
jupyter nbconvert --to python --output-dir=scripts/ *.ipynb

# run python script to spit, train, and evaluate the model (ran in sequential order)
python scripts/0.split_data.py
python scripts/1.train_models.py
python scripts/2.evaluate_models.py
