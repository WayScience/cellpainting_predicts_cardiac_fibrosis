# 1. Preprocessing CFReT Data

In this module, we present our pipeline for preprocessing the CFReT pilot data.

## Illumination Correction

To correct for illumination issues within the CFReT pilot data, we use the BaSiC method that was established in an article by [Peng et al.](https://doi.org/10.1038/ncomms14836).
We specifically use the Python implementation of this method, called [PyBaSiC](https://github.com/peng-lab/BaSiCPy).

Illumination correction is an important step in cell image analysis pipelines as it helps with downstream processes like segmentation (more accuracy in segmenting shape and identifying objects to segment) and feature extraction (accurate measurements in intensity, texture, etc.).

Being able to visualize illumination errors can be hard for some datasets.
As shown in Figure 1 below, the images are brightened using [Fiji](https://imagej.net/software/fiji/) to be able to see the variation in illumination across the image (e.g vignetting).

![Raw_to_Bright_fig](example_figs/Raw_to_Bright_fig.png)

*Figure 1. Image Comparison: Raw to Brightened. This figure displays the images for each channel in the same well/frame. The channel metadata (red) and Cell Painting assay stain for each channel (blue) is shown to the left of the images.*

After using PyBaSiC (or BaSiCPy) to correct for illumination, you can see the difference made on the images through brightening them, as shown in Figure 2.

![Raw_to_Corrected_fig](example_figs/Raw_to_Corrected_fig.png)

*Figure 2. Image Comparison: Brightened Raw Images to Brightened Corrected Images. This figure shows the difference in illumination when the raw and corrected images are brightened.*

## Step 1: Install PyBaSiC

Clone the repository into 1_preprocess_data/ with 

```console
git clone https://github.com/peng-lab/PyBaSiC.git
git checkout f3fcf1987db47c4a29506d240d0f69f117c82d2b
```

**Note:** This implementation does not have package support which means that it can not be imported as you normally would. 
To correct for this, use this line of code within your "Importing Libraries" cell to be able to use the functions within the 
[notebook](1.preprocessing-data/illumcorrect-data.ipynb).

```console
import sys
sys.path.append("./PyBaSiC/")
import pybasic
```
