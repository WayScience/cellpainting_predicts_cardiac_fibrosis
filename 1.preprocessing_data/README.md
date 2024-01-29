# 1. Preprocessing CFReT Data

In this module, we present our pipeline for preprocessing the CFReT pilot data.

## Whole Image Quality Control (QC)

Based on current research, the method of quality control for whole images involves some sort of manually annotation.
Manual annotations take a lot of time and is hard to implement with the large scale of images per plate.

In our method we are implementing (temporarily dubbed Jenna's QC method) for whole image QC involves running a [whole_image_qc pipeline](./pipelines/whole_image_qc.cppipe) in CellProfiler to extract blur and saturation metrics to assess image quality into a CSV file called `Image.csv`.
After running the pipeline, we evaluate the QC metrics for blur and saturation to determine necessary thresholds.
Those thresholds are then added into the illumination correction pipeline to flag images that of poor quality.

### Blur metric: PowerLogLogSlope

To assess if there significant impact on the image set from blur, we use the PowerLogLogSlope metric calculated for each channel. 
The definition of this metric can be found in the [CellProfiler manual](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.6/modules/measurement.html#id8), but we have found that it is not easily interpretable.

Our definition of the metric is as follows:

> A blurry image will have a PowerLogLogSlope value closer to 0 (less negative) while good quality images will have a more negative value. The range of values when outputted straight from CellProfiler can range from any values less than 0 (negative values).

To evaluate if blur is impacting the dataset, we look at the distribution of values per channel as a density plot. 
We expect to see a bump in the distribution near 0 if there is a significant portion of images with blur.
So far in CFReT, blur is not impacting the dataset.

### Saturation metric: PercentMaximal

To assess if there significant impact on the image set from large and highly saturated smudges/artifacts, we use the PercentMaximal metric calculated for each channel. 
The definition of this metric can be found in the [CellProfiler manual](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.6/modules/measurement.html#id8).

Due to the significantly skewed distribution of variables, we currently do not have a visualization of this metric.
We use z-scoring to identify outliers that are 2 standard deviations above the mean since we are looking for abnormally saturated images.

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
