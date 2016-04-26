import pdb
import os, sys
import numpy as np
from skimage.segmentation import slic
from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries, relabel_sequential
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imread

lib_path = os.path.abspath("/home/sheng/workspace/OpenPV/pv-core/python/")
sys.path.append(lib_path)
from pvtools import *

lib_path = os.path.abspath("/home/sheng/workspace/OpenPV/pv-core/python/tmp/")
sys.path.append(lib_path)
from writePvpFile import *


if __name__ == "__main__":
    outputDir = "/home/sheng/mountData/superpixel/train/"

    leftReconObj = readpvpfile(outputDir + "a1_LeftRecon.pvp");

    origSegObj = readpvpfile("/home/sheng/workspace/superpixelDepth/output/kittiSeg.pvp");

    segDepthObj = readpvpfile(outputDir + "a6_segDepth.pvp");
    segObj = readpvpfile(outputDir + "a4_segment.pvp");

    leftReconData = leftReconObj[10].values
    segDepthData = segDepthObj[10].values
    segData = segObj[10].values
    origSegData = origSegObj[10].values

    #Use jet colormap, where value of vmin gets set to black (for DNC regions)
    colormap = cm.get_cmap('jet')
    colormap.set_under('black')

    f, ax = plt.subplots(3, 1, sharex=True)

    ax[0].imshow(leftReconData[:, :, 0], cmap='gray')
    ax[0].set_title("Orig")
    ax[1].imshow(origSegData[:, :, 0], cmap='gray')
    ax[1].set_title("Seg depth")
    ax[2].imshow(segDepthData[:, :, 0], cmap=colormap, vmin=.0001)
    ax[2].set_title("Seg depth")

    plt.show()


    pdb.set_trace()




    ##Use jet colormap, where value of vmin gets set to black (for DNC regions)
    #colormap = cm.get_cmap('jet')
    #colormap.set_under('black')

    #f, ax = plt.subplots(4, 1, sharex=True)

    #ax[0].imshow(img)
    #ax[0].set_title("Orig")
    #ax[1].imshow(mark_boundaries(img, segments))
    #ax[1].set_title("SLIC")
    #ax[2].imshow(depth, cmap=colormap, vmin=.0001)
    #ax[2].set_title("Orig depth")
    #axx = ax[3].imshow(segDepth, cmap=colormap, vmin=.0001)
    #ax[3].set_title("Seg depth")

    #f.subplots_adjust(right=.8)
    #cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    #f.colorbar(axx, cax=cbar_ax)

    #plt.figure()
    #axx = ax[3].imshow(segDepth, cmap=colormap, vmin=.0001)






