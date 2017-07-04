SeAM
=====

(SEgmentation And Measurement)

## nucleiSeg ##
**a script for nucleus segmentation from a DAPI-stained worm gonad image**

nucleiSeg.py is the source code of a snippet I wrote for labeling out nuclei from a DAPI-stained worm gonad image.

This script takes a image file as the input and segments nuclei out according to a series of user-defined parameters (for instance, diameter and shape factor etc). The final segmentation result would be recorded in the user-defined CSV file.

The script makes use of the powerful Python image analysis toolkit scikit-image for this task. The workhorse underneath the cover is the watershed algorithm.

Usage:
`$ python nucleiSeg.py <image filename> <target CSV file>`

The Python version I was using for this script is 3.5.
