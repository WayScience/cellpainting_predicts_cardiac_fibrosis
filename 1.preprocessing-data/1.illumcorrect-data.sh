#!/bin/bash
jupyter nbconvert --to python illumcorrect-data.ipynb
python illumcorrect-data.py