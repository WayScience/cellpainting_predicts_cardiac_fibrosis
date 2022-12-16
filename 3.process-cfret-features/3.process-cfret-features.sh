#!/bin/bash

# convert jupyter notebook to a .py file once all variable changes are made
jupyter nbconvert --to python extract_sc_features.ipynb

# run notebook through Python
python extract_sc_features.py
