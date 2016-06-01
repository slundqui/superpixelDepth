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
   colormap.set_bad('black')
   img = np.ma.masked_where(img == 0, img)
   f = plt.imshow(img, cmap=colormap)
   plt.colorbar()
   plt.show()

def plotDepth(image, segments, labels, gt, est, outFilename):
   gtImg = fillSegments(segments, gt, labels)
   estImg = fillSegments(segments, est, labels)
   gtImg = np.ma.masked_where(gtImg == 0, gtImg)
   estImg = np.ma.masked_where(estImg == 0, estImg)

   #Set range to that of the ground truth
   vmax = np.max(gtImg)
   vmin = np.min(gtImg)

   colormap = cm.get_cmap('jet')
   colormap.set_bad('black')

   f, ax = plt.subplots(3, 1, sharex=True)
   ax[0].imshow(image)
   ax[0].set_title("Image")
   ax[1].imshow(gtImg, cmap=colormap, vmax = vmax, vmin = vmin)
   ax[1].set_title("GT")
   axx=ax[2].imshow(estImg, cmap=colormap, vmax = vmax, vmin = vmin)
   ax[2].set_title("EST")

   f.subplots_adjust(right=.8)
   cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
   f.colorbar(axx, cax=cbar_ax)
   plt.savefig(outFilename)

   plt.close(f)
