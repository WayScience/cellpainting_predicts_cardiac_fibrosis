# 1. Preprocessing CFReT Data

In this module, we present our pipeline for preprocessing the CFReT pilot data.

## Illumination Correction (IC)

To correct for illumination issues within the CFReT pilot data, we use the Background method for all plates from the [CellProfiler illumination correction modules](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.4/modules/imageprocessing.html#correctilluminationapply).

To confirm that IC was working on the images, we adjusted the brightness/contrast in CellProfiler to see if the contrast between objects and background increased and the illumination looked more even. 

## Calculate and save corrected images for all plates

To perform illumination correction and save the corrected images, run the [cfret_ic.sh](./cfret_ic.sh) file using the below command:

```bash
# activate the conda env for the repo
conda activate cfret_data
# start the parallelized IC run
source cfret_ic.sh
```

For my computer which is a Pop_OS LTS 22.04 system with an AMD Ryzen 7 3700X 8-Core Processor, it took this pipeline about 50 minutes to process all 4 plates parallel.

