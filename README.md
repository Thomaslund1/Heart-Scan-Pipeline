# Heart Scan Data Reduction Pipeline
## 17-Dec-2025 

## Segmentation
This program segments the image stack using the new difference of medians filter to extract medium sized features from the backround and reduce noise.
USAGE:
 > python segmentation.py {input directory} {output directory} {small window size = 3} {large window size = 17}

input directory : str
    The path to the folder containing the tiff files, may be absolute or relative to current working directory
output directory : str
    Path to where the processed masks should go
small window size : int
    Size of the noise-reducing median filter, 2 or 3 should work for most cases
large window size : int
    Size of the backround eliminating median filter, large values take more compute, but allow for larger feature detection

## TiffToRaw
A combination program that converts the above binary masks to a single .raw file. Also allows basic 3d model manipulation like downsampling, kernaled erosion/dilation, and rudimentary path briding. 
USAGE:
 > python tiffToRaw.py {inputFolder} {outputFolder} {processingMethod="morphological"}

inputFolder : str
    Path to the binary mask folder
outputFolder : str
    Where the output the .raw file
processingMethod : {'morphological' / 'adt'}
    Which method to apply to clean the data, further paramaters may be adjusted in the program
