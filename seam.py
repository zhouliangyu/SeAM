import sys
import gonadSeg
from matplotlib import pyplot as plt
from skimage import io




def glance(img):
    oriImg = io.imread("../images/gonadDapi.jpg")
    fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True)
    ax = axes.ravel()
    ax[0].imshow(oriImg, cmap = plt.cm.gray)
    ax[0].axis("off")
    ax[1].imshow(img, cmap=plt.cm.gray)
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()



if len(sys.argv) != 2:
    sys.exit("Usage: $ python seam.py <image filename>") 
imgfile=sys.argv[1]
print("Filename: ", imgfile)

nuclei = gonadSeg.segNucleiOut(imgfile) 



glance(nuclei)
