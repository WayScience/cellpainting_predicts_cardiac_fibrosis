# 1. Preprocessing CFReT Data

In this module, we present our pipeline for preprocessing the CFReT pilot data.

## Illumination Correction (IC)

To correct for illumination issues within the CFReT pilot data, we use the Background method for all plates from the [CellProfiler illumination correction modules](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.4/modules/imageprocessing.html#correctilluminationapply).

To confirm that IC was working on the images, we adjusted the brightness/contrast in CellProfiler to see if the contrast between objects and background increased and the illumination looked more even. 

## Calculate and save corrected images for all plates

To perform illumination correction and save the corrected images, you will need to:

### Step 1: Open the CellProfiler GUI

To open the CellProfiler GUI, run the code block below:

```bash
# activate the conda env for the repo
conda activate cfret_data
# call CellProfiler to start the GUI
cellprofiler
```

### Step 2: Open the pipeline in the GUI

Drag the `illum.cppipe` file into the GUI to open the pipeline.
Once the pipeline is open, go to the `Images` module and clear the file list.
Since Cellprofiler needs absolute paths, you will need to drag the "Images" folder from your local machine in this module to make sure Cellprofiler can find the plates with the images.

### Step 3: Start pipeline run

Once the `Images` path is correct, you can press `Start Analysis` to run the pipeline on all plates simultaneously.

For my computer which is a Pop_OS LTS 22.04 system with an AMD Ryzen 7 3700X 8-Core Processor, it took this pipeline about 1 hour to run a total of 11,280 images (between three plates).



