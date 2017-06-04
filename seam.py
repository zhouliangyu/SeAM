from skimage import io
from skimage.color import rgb2gray, gray2rgb
from skimage import img_as_ubyte
from matplotlib import pyplot as plt
from skimage.filters import threshold_sauvola
import numpy as np
from skimage.morphology import opening, closing, disk, watershed
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter as circle



def quiDisp(*img, c_map = plt.cm.gray):
    totImgNum = len(img)
    fig, ax = plt.subplots(nrows = totImgNum, sharex = True,
            sharey = True)
    ax[0].imshow(img[0], cmap = plt.cm.gray)
    ax[0].axis("off")
    for i in range(totImgNum-1):
        ax[i+1].imshow(img[i+1], cmap = c_map)
        ax[i+1].axis("off")
    plt.tight_layout()
    plt.show(block=False)


def loadImg(imgFileName):
    img = io.imread(imgFileName)
    img = rgb2gray(img)
    img = img_as_ubyte(img)
    return img

def thresImg(img, window_size = 51, k = 0):
    thres = threshold_sauvola(img, window_size = window_size, k = k)
    return img > thres 

def idenNucl(img, min_dist = 25):
    dist = ndi.distance_transform_edt(img)
    local_maxi = peak_local_max(dist, indices = False,
            min_distance = min_dist, labels = img) 
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-dist, markers, mask = img)
    return labels


fileName = "../images/gonadDapi.jpg"

oriImg = loadImg(fileName)
img = oriImg.copy() 
img = thresImg(img)
img = opening(img, selem = disk(1)) 


# start iter

img = closing(img, selem = disk(1))
img = ndi.morphology.binary_fill_holes(img)

labeled = idenNucl(img, min_dist = 25)
totalInd = labeled.max()
allReg = regionprops(labeled)
tempImg = gray2rgb(oriImg) 
fig, ax = plt.subplots(nrows = 3, sharex = True, sharey = True)
ax[0].imshow(oriImg, cmap = plt.cm.gray)
ax[1].imshow(labeled, cmap = "tab20")



for i in range(totalInd):
    if allReg[i].equivalent_diameter <= 25 or \
        allReg[i].equivalent_diameter >= 80:
            continue
    else:
        min_row, min_col, max_row, max_col = allReg[i].bbox
        rowSpan = max_row - min_row
        colSpan = max_col - min_col
        cirRadi = int(max(rowSpan, colSpan)/2)
        for drawCir in range(5):
            rr, cc = circle(int(allReg[i].centroid[0]),
                    int(allReg[i].centroid[1]), 
                    cirRadi + drawCir - 2)
            tempImg[rr, cc, 1:3] = 0



ax[2].imshow(tempImg)
plt.show(block=False)    




