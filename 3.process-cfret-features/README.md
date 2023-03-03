# 3. Processing Extracted Single Cell and Image Features from CellProfiler

In this module, we present our pipeline for processing outputted `.sqlite` files with single cell and image features from CellProfiler.
The processed features are saved into compressed `.csv.gz` files for use during statistical analysis for each plate (except for image features).
We are only extracting image features from the second plate (e.g., Factin_Adjusted plate) as we are planning on continuing with this Cell Painting protocol for further plates.

## Pycytominer

We use [Pycytominer](https://github.com/cytomining/pycytominer) to perform the aggregation, merging, and normalization of the CFReT single cell features.

For more information regarding the functions that we used, please see [the documentation](https://pycytominer.readthedocs.io/en/latest/pycytominer.cyto_utils.html#pycytominer.cyto_utils.cells.SingleCells.merge_single_cells) from the Pycytominer team.

### Normalization

CellProfiler features can display a variety of distributions across cells.
To facilitate analysis, we standardize all features (z-score) to the same scale.

### Feature Selection

Pycytominer will use specified operations to perform feature selection and remove features from the dataframe that are not significant.

---

## Step 1: Setup Processing Feature Environment

### Step 1a: Create Environment

Make sure you are in the `3.processing_CP_features` directory before performing the below command.

```bash
# Run this command in terminal to create the conda environment
conda env create -f 3.preprocessing_CFReT_features.yml
```

### Step 1b: Activate Environment

```bash
# Run this command in terminal to activate the conda environment
conda activate 3.process-cfret-features
```

## Step 2: Extract Single Cell and Image Features

Using the code below, run the notebook to extract and normalize single cell features from CellProfiler.

```bash
# Run this script in terminal
bash 3.process-cfret-features.sh
```
