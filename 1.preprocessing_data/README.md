# 1. Preprocessing CFReT Data

In this module, we present our pipeline for preprocessing the CFReT pilot data.

## Whole Image Quality Control (QC)

Based on current research, the method of quality control for whole images involves some sort of manually annotation.
Manual annotations take a lot of time and is hard to implement with the large scale of images per plate.

In our method we are implementing (temporarily dubbed Jenna's QC method) for whole image QC involves running a [whole_image_qc pipeline](./pipelines/whole_image_qc.cppipe) in CellProfiler to extract blur and saturation metrics to assess image quality into a CSV file called `Image.csv`.
After running the pipeline, we evaluate the QC metrics for blur and saturation to determine necessary thresholds.
Those thresholds are then added into the illumination correction pipeline to flag images that of poor quality.

### Assess QC metrics and create QC report for whole images

Separate to performing IC, you can run the QC pipeline, assess the metrics, and create a whole image QC report by running the command below:

```bash
# start the QC processing and reporting run
source cfret_qc_report.sh
```

## Illumination Correction (IC)

To correct for illumination issues within the CFReT pilot data, we use the Background method for all plates from the [CellProfiler illumination correction modules](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.4/modules/imageprocessing.html#correctilluminationapply).

To confirm that IC was working on the images, we adjusted the brightness/contrast in CellProfiler to see if the contrast between objects and background increased and the illumination looked more even. 

### Calculate and save corrected images for all plates

>[!Note]
>The above QC notebooks do NOT need to be run prior to running IC. In the IC pipeline, we already have whole image QC thresholds set to use for processing the data.

To perform illumination correction and save the corrected images, run the [cfret_ic.sh](./cfret_ic.sh) file using the below command:

```bash
# start the parallelized IC run
source cfret_preprocessing.sh
```

The pipeline was tested on a Pop_OS LTS 22.04 system with an AMD Ryzen 7 3700X 8-Core Processor taking ~50 minutes to process all 4 plates parallel.
