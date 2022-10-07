#!/bin/bash

# convert jupyter notebook to a .py file once all variable changes are made
jupyter nbconvert --to python example_cp.ipynb

# run notebook through Python
python example_cp.py
