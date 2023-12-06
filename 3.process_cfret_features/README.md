# 3. Processing Extracted Single Cell and Image Features from CellProfiler

In this module, we present our pipeline for processing outputted `.sqlite` files with single cell and image features from CellProfiler.
The processed features are saved into `parquet` files, which are used for analysis for each plate.

## CytoTable

We use [CytoTable] to convert the CellProfiler SQLite outputs into merged single-cell parquet files, where features from each compartment (e.g., nuclei, cells, and cytoplasm) are combined into one row per single-cell.
The exact function we used is called `convert`, which you can find more info on in the [documentation](https://cytomining.github.io/CytoTable/python-api.html#module-cytotable.convert).

## Pycytominer

We use [pycytominer](https://github.com/cytomining/pycytominer) to perform the aggregation, merging, and normalization of the CFReT single cell features.

For more information regarding the functions that we used, please see the [documentation](https://pycytominer.readthedocs.io/en/latest/pycytominer.cyto_utils.html#pycytominer.cyto_utils.cells.SingleCells.merge_single_cells) from the pycytominer team.

### Normalization

CellProfiler features can display a variety of distributions across cells.
To facilitate analysis, we standardize all features (z-score) to the same scale.

### Feature Selection

Pycytominer will use specified operations to perform feature selection and remove features from the data frame that are not significant.

---

## Perform processing on CellProfiler features

Using the code below, run the notebook to perform all processing steps to generate single-cell features from CellProfiler.

```bash
# Make sure you are in the 3.process_cfret_features module
cd 3.process_cfret_features
# Run this script in terminal
source process_data.sh
```
