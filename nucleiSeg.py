import sys
import math
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from skimage.filters import threshold_sauvola
from skimage.morphology import disk, opening, closing, watershed
from skimage.measure import regionprops
from scipy import ndimage as ndi
from skimage.draw import circle as circle_solid
from skimage.draw import circle_perimeter as circle_empty

# function for watershed segmentation
def nucleiIdentify(img, min_dist = 25):
    dist = ndi.distance_transform_edt(img)
    local_maxi = peak_local_max(dist, indices = False,
            min_distance = min_dist, labels = img)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-dist, markers, mask = img)
    return labels

# function for judging nucleus shape (near-circular or rod-like)
def compareRadius(labObj, rad_thres = 0.35):
    nearCircular = False
    min_row, min_col, max_row, max_col = labObj.bbox
    equiRadi = labObj.equivalent_diameter / 2
    rowSpan = max_row - min_row
    colSpan = max_col - min_col
    nucleusRadius = max(rowSpan, colSpan)/2
    radErr = abs(equiRadi - nucleusRadius) / min(equiRadi, nucleusRadius)
    if radErr <= rad_thres:
        nearCircular = True
    return nearCircular, math.ceil(max(equiRadi, nucleusRadius))

# read commandline arguments
if len(sys.argv) != 3:
    print ("Usage: $ python nucleiSeg.py <image filename> <target CSV file>")
    exit()
imgFileName = sys.argv[1]
tgtFileName = sys.argv[2]
img = io.imread(imgFileName)
img = rgb2gray(img)
img = img_as_ubyte(img)
oriImg = img.copy()

# thresholding : window_size = number of pixels in each nuclei
userWinSize = 51
thresVal = threshold_sauvola(img, window_size = userWinSize, k = 0)
img = img > thresVal

# denoise the image using morphology opening
userDenoiseStr = disk(1)
img = opening(img, selem = userDenoiseStr) 
initialFG = closing(img, selem = disk(1)).sum() # for computing the percentage

nucleiList = []
iterCount = 0
iterThres = 0.99      # percentage of foreground: for terminating the iteration
maskedPercentage = 0  # work together with iterThres
while True:
# closeStrength determines which gaps (size) would be closed. Keep small.
    closeStrength = disk(min(5, iterCount // 8 + 1))
    img = closing(img, selem = closeStrength)
    img = ndi.morphology.binary_fill_holes(img)
# seperateDist determines how far two centriod of nuclei should be.
    seperateDist = max(20, 30 - iterCount // 3)
    labeled = nucleiIdentify(img, min_dist = seperateDist)
    totalIdentified = labeled.max()
    allRegions = regionprops(labeled)
    imgMask = np.ones(img.shape)
    for i in range(totalIdentified):
        currRegDiameter = allRegions[i].equivalent_diameter
# define nucleus size boundaries
        nucleiMinBoundary = max(20, (26 - iterCount // 4))
        nucleiMaxBoundary = min(65, (62 + iterCount // 4))
        if  currRegDiameter <= nucleiMinBoundary or currRegDiameter >= nucleiMaxBoundary:
            continue # descard nuclei fall beyond size boundaries
        else:
            crescentRatio = 0.35 # deviation of crescent shape from a circle
            nearCircular, nucleusRadius = compareRadius(allRegions[i], crescentRatio)
            if not nearCircular:
                continue # descard rod-like regions
            else:
                cirX = int(allRegions[i].centroid[0])
                cirY = int(allRegions[i].centroid[1])
                nucleiList.append((cirX, cirY, nucleusRadius))
                rr, cc = circle_solid(cirX, cirY, nucleusRadius)
                imgMask[rr, cc] = 0 # mask identified nuclei
    maskedPercentage += (imgMask == 0).sum() / initialFG
    img = img * imgMask
    iterCount += 1
    print("Iteration {}: {}% processed, picked {} nuclei.".format(
        iterCount, round(maskedPercentage * 100), len(nucleiList)))
    if maskedPercentage >= iterThres:
        break
# write results to file
finalResult = pd.DataFrame(nucleiList, columns = ['X', 'Y', 'Radius'])
finalResult.to_csv(tgtFileName, index = False)
print("Final segmentation is shown for your reference (close the image window to quit).")
oriImg = oriImg // 2
for cirX, cirY, nucleusRadius in nucleiList:
    for drawCir in range(3):
        rr, cc = circle_empty(cirX, cirY, nucleusRadius + drawCir - 1)
        oriImg[rr, cc] = 255
plt.imshow(oriImg)
plt.show()
