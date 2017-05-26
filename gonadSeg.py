# first: segmentation of nuclei
from skimage import io
from skimage import color
from skimage import img_as_ubyte
from skimage.filters import threshold_sauvola as sauvola
from skimage.morphology import closing, disk, opening, erosion, dilation
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes


def loadImgfile(imgfile, gaussianValue = 2):
    img = io.imread(imgfile)
    img = color.rgb2gray(img)
#     img = gaussian_filter(img, gaussianValue) 
    img = img_as_ubyte(img)
    return img

def localThres(img, windowSize = 51):
    threshed = sauvola(img, windowSize)
    resultImg = img > threshed
    return resultImg

def deNoise(img, selemO = disk(2), selemC = disk(2)):
    opened = opening(img, selemO)
    closeIt = closing(img, selemC)
    return closeIt 

def fillIt(img):
    filled = binary_fill_holes(img)
    return filled

# def fillByOpen(img, iterCycle = 1, selemIter = 5):
#     resultImg = img
#     for i in range(iterCycle):
#         j = 1
#         while (j <= selemIter):
#             resultImg = opening(resultImg, disk(j))
#             j += 1
#     return resultImg

def fillByClose(img, selemC = disk(5)):
    resultImg = closing(img, selemC)
    return resultImg


def segNucleiOut(imgfile):
    img = loadImgfile(imgfile)
    img = localThres(img) 
    img = deNoise(img) 
#     img = fillIt(img) 
#     img = fillByOpen(img) 
#     img = fillByClose(img)




    return img
