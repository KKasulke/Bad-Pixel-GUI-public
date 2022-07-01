# Bad-Pixel-master
GUI for detection and correction of bad pixels in image files

## Installation
* import "anaconda_python_environment.yaml" in anaconda to install all required libraries

## Limitations
* image resolution: ai model training and ai image correction is only possible for images with resolutions of 256x256 or 512x512, the resolution of the bad pixel map has to match with the resolution of the training images, the ai-model and the images which shall be corrected
* only .png-files are supported with AI - Partial Convolution image correction

## Possible improvements
* check for possible errors (for example wrong image resolutions or no ai-model selection) before starting image correction with partial convolution
