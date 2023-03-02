#!/bin/bash

# convert single cell jupyter notebook to a .py file once all variable changes are made
jupyter nbconvert --to python extract_sc_features.ipynb

# run notebook through Python
python extract_sc_features.py

# convert single cell jupyter notebook to a .py file once all variable changes are made
jupyter nbconvert --to python extract_image_features.ipynb

# run notebook through Python
python extract_image_features.py
