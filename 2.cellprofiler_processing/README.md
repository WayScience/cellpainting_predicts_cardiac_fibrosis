# 2. CellProfiler Segmentation and Feature Extraction

In this module, we present our pipeline for segmentation and feature extraction using [CellProfiler (CP)](https://cellprofiler.org/)

## Segmentation

To run segmentation, we use the standard approach used in CellProfiler for Cell Painting images (see https://github.com/gigascience/paper-bray2017/tree/master/pipelines and https://github.com/broadinstitute/imaging-platform-pipelines)

This standard approach is as follows:

- **IdentifyPrimaryObjects:** Identify nuclei from the `d0` channel images (discarding ones that touch the edges of the image). This creates a group for an object called `Nuclei`.

- **IdentifySecondaryObjects:** Identify whole cells using the nuclei from the previous module as a base from the `d4` channel (stained for F-actin) images (discarding edge cells). These whole cells make up the object group called `Cells`.

- **IdentifyTeritaryObjects:** Identify cytoplasm by subtracting out the "smaller identified objects" (`Nuclei`) from the "larger identified objects" (`Cells`). 

## Feature Extraction

To run feature extraction, the modules used are based on the ones from the [NF1 Schwann Cell CellProfiler pipeline](https://github.com/WayScience/NF1_SchwannCell_data/tree/main/CellProfiler_pipelines).
The modules used include:

- **MeasureColocalization**
- **MeasureGranularity**
- **MeasureObjectIntensity** 
- **MeasureImageIntensity**
- **MeasureObjectNeighbors**
- **MeasureObjectIntensityDistribution**
- **MeasureObjectSizeShape**
- **MeasureTexture**
- **MeasureImageQuality**

For more information on these modules and the parameters within me, please reference the NF1 Schwann Cell project documentation above for the CellProfiler pipeline.
More modules can be added based on the needs of the project.

## ExportToDatabase Module

Within this module, the features for each object (e.g. nuclei, cells, and cytoplasm) are exported into a SQLite file.

Specifically, this module is set to create one table per object type, which is the expected format for the file for the next step in the pipeline (e.g. `preprocessing features`).

--- 
## Run CellProfiler analysis on each plate

To run CellProfiler analysis on each plate, run the [cp_analysis.ipynb notebook](cp_analysis.ipynb) using the code below:

```bash
# change directory to the module with the bash script
cd 2.cellprofiler_processing/
# Run this script in terminal
source cfret_analysis.sh
```

CellProfiler runs on CPU in sequential order, and it took approximately:
- ~14 hours to run plate 1 (`localhost220512140003_KK22-05-198`) 
- ~8 hours to run plate 2 (`localhost220513100001_KK22-05-198_FactinAdjusted`)
- ~10 hours to run on plate 3 (`localhost230405150001`)

This totals to **32 hours** to run all plates.
The analysis was run on a Linux-based machine running Pop_OS! LTS 22.04 with an AMD Ryzen 7 3700X 8-Core Processor.
