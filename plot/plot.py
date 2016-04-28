import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from segment.segment import segmentDepth, calcSegments, fillSegments
import pdb

def plotLoss(lossVals, outFilename):
   plt.plot(lossVals)
   plt.savefig(outFilename)

def plotImg(segments, labels, vals):
   img = fillSegments(segments, vals, labels)
   colormap = cm.get_cmap('jet')
   colormap.set_under('black')
   plt.imshow(img, cmap=colormap, vmin=.0001)
   plt.show()

def plotDepth(segments, labels, gt, est, outFilename):
   gtImg = fillSegments(segments, gt, labels)
   estImg = fillSegments(segments, est, labels)

   colormap = cm.get_cmap('jet')

   f, ax = plt.subplots(2, 1, sharex=True)
   ax[0].imshow(gtImg, cmap=colormap)
   ax[0].set_title("GT")
   axx=ax[1].imshow(estImg, cmap=colormap)
   ax[1].set_title("EST")

   f.subplots_adjust(right=.8)
   cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
   f.colorbar(axx, cax=cbar_ax)
   plt.savefig(outFilename)

