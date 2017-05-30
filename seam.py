from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.filters import threshold_niblack
from skimage import img_as_ubyte
from skimage.morphology import opening, dilation
from skimage.morphology import disk, watershed
from scipy import ndimage as ndi
import numpy as np
from skimage import io
from skimage.feature import peak_local_max



class GonadImg:

    def __init__(self, imgFilename):
        self.srcFile = imgFilename
        self.img = img_as_ubyte(rgb2gray(io.imread(imgFilename)))

    def quickDisp(self):
        plt.imshow(self.img, cmap = plt.cm.gray)
        plt.show(block = False)
    
    def thres(self, thresParam = 51, inPlace = False):
        if thresParam % 2 == 0:
            thresParam += 1
        thresBdry = threshold_niblack(self.img, window_size = thresParam, k = 0)
        if inPlace:
            self.img = self.img > thresBdry
        return self.img > thresBdry

    def deNoise(self, selem = disk(1), inPlace = False):
        if inPlace:
            self.img = opening(self.img, selem)
        return opening(self.img, selem)

    def fillGap(self, selem = disk(1), inPlace = False):
        if inPlace:
            self.img = dilation(self.img, selem)
        return dilation(self.img, selem)

    def fillHole(self, inPlace = False):
        if inPlace:
            self.img = ndi.binary_fill_holes(self.img)
        return ndi.binary_fill_holes(self.img)

    def nucLabel(self, inPlace = False):
        labeled, labelNum = ndi.label(self.img)
        if inPlace:
            self.img = labeled 
        return labeled, labelNum




def batchDisp(*imageList, cmap=plt.cm.gray):
    imgNum = len(imageList)
    fig, ax = plt.subplots(nrows = imgNum, ncols = 1, 
            sharex = True, sharey = True)
    ax = ax.ravel()
    for i in range(imgNum):
        ax[i].imshow(imageList[i], cmap = cmap)
        ax[i].axis("off")
    plt.tight_layout()
    plt.show(block = False)



testFile = "../images/gonadDapi.jpg"
gonad = GonadImg(testFile)
gonad.thres(inPlace = True)

oriImg = gonad.deNoise()

gonad.deNoise(inPlace = True)
gonad.fillGap(inPlace = True)
gonad.fillHole(inPlace = True)

test, tempNum = gonad.nucLabel()
batchDisp(gonad.img, test)


# gonad.quickDisp()
