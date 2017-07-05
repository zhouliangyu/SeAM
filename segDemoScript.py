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
from skimage.draw import circle as solid_circle
from skimage.draw import circle_perimeter as circle
import math, time

# functions for segmentation

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

def chkStat(promStr, img, c_map = plt.cm.gray):
    userInp = input("Check {} result? (y for yes) ".format(promStr))
    if userInp == "y":
        quiDisp(oriImg, img, c_map = c_map)
        input("Press enter to continue.")
        plt.close("all")
    else:
        print("Skip checking.")

def compRadi(labObj, rad_thres = 0.35):
    nucQual = False
    min_row, min_col, max_row, max_col = labObj.bbox
    equiRadi = labObj.equivalent_diameter / 2
    rowSpan = max_row - min_row
    colSpan = max_col - min_col
    cirRadi = max(rowSpan, colSpan)/2
    radErr = abs(equiRadi - cirRadi) / min(equiRadi, cirRadi)
    if radErr <= rad_thres:
        nucQual = True
    return nucQual, math.ceil(max(equiRadi, cirRadi))
    

    


debugStat = False
eachIter = False
fileName = "../images/gonadDapi.jpg"
oriImg = loadImg(fileName)
img = oriImg.copy() 
img = thresImg(img)
if debugStat:
    chkStat("thresholding", img)
img = opening(img, selem = disk(1)) 
if debugStat:
    chkStat("denoising", img)


# start iter

nucList = [] 
iterCnt = 0
imgBef = closing(img, selem = disk(1))
iterThres = 0.99
totMasked = 0
while True:
    img = closing(img, selem = disk(min(5, iterCnt // 8 + 1)))
    img = ndi.morphology.binary_fill_holes(img)
    if debugStat:
        chkStat("hole-filling", img)
    labeled = idenNucl(img, min_dist = max(20, 30 - iterCnt // 3))
    if debugStat:
        chkStat("object-labeling", labeled, c_map = "tab20")
    totalInd = labeled.max()
    allReg = regionprops(labeled)
    tempImg = gray2rgb(oriImg) 
    imgMask = np.ones(img.shape)

    for i in range(totalInd):
        if allReg[i].equivalent_diameter <= max(20, (26 - iterCnt // 4)) or \
            allReg[i].equivalent_diameter >= min(65, (62 + iterCnt // 4)):
            continue
        else:
            nucQual, cirRadi = compRadi(allReg[i])
            if not nucQual:
                continue
            else:
                cirX = int(allReg[i].centroid[0])
                cirY = int(allReg[i].centroid[1])
                nucList.append((cirX, cirY, cirRadi))
                rr, cc = solid_circle(cirX, cirY, cirRadi)
                imgMask[rr, cc] = 0
                for drawCir in range(3):
                    rr, cc = circle(cirX, cirY, cirRadi + drawCir - 1)
                    tempImg[rr, cc, 0:2] = 255

    if eachIter:
        fig, ax = plt.subplots(nrows = 2, sharex = True, sharey = True, 
                figsize = (15, 7))
        ax[0].imshow(labeled, cmap = "tab20")
        ax[0].axis("off")
        ax[1].imshow(tempImg)
        ax[1].axis("off")
        plt.tight_layout()
        plt.show(block = False)    
        input("press enter to continue")
        plt.close("all")
    totMasked += (imgMask == 0).sum() / imgBef.sum()
    img = img * imgMask
    iterCnt += 1
    print("No. {} iteration, processed {}%, in total picked {} nuclei.".format(
        iterCnt, round(totMasked * 100), len(nucList)))
    if totMasked >= iterThres:
        break

if debugStat:
    print("Final list: ", nucList)

oriImg = oriImg // 2
for cirX, cirY, cirRadi in nucList:
    for drawCir in range(3):
        rr, cc = circle(cirX, cirY, cirRadi + drawCir - 1)
        oriImg[rr, cc] = 255

plt.figure(figsize=(15, 3.5))
plt.imshow(oriImg)
plt.axis("off")
plt.tight_layout()
plt.show(block=False)
    



