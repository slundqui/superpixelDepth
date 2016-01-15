import cv2
import pdb
import numpy as np
from skimage.segmentation import slic
from skimage.future import graph
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage import data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

imageFilename = "input/img_frame_000021_10.png"
#Number of segments
slic_nsegments = 400
#Balances color and space proximity
#Higher gives more uniform segments
slic_compactness = 10
#Width of gaussian smoothing kernel for preprocessing
slic_sigma = 1

#Number of segments
seeds_nsegments = 400
#Number of block levels, more levels, more accurate
seeds_nlevels = 4
# 3x3 shape smoothing
seeds_prior = 2
#Histogram of color bins for energy function
seeds_nhistbins = 5
#Number of pixel level iterations
seeds_niterations = 10

img = cv2.imread(imageFilename)

height,width,channels = img.shape

#SEEDS segmentation
seedsObj = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, seeds_nsegments, seeds_nlevels, seeds_prior, seeds_nhistbins)

seedsObj.iterate(img, seeds_niterations)

#Get results
segments_seeds = seedsObj.getLabels()


#matplotlib color channels are RGB, whereas cv is BGR. Flip dimensions
img = img[:, :, ::-1]

assert(img != None)

#img = img_as_float(img)

#SLIC segmentation
segments_slic = slic(img, n_segments=slic_nsegments, compactness=slic_compactness, sigma=slic_sigma)

#NCuts segmentation
#Initial labels are one per pixel
#initSegments_ncuts = np.reshape(range(height*width), [height, width])

fsegments_slic = segments_slic

#Using slic segments
g = graph.rag_mean_color(img, segments_slic, mode='similarity')
segments_ncuts = graph.cut_normalized(segments_slic, g)

#Plot
f, ax = plt.subplots(4, 1, sharex=True)

ax[0].imshow(img)
ax[0].set_title("Orig")
ax[1].imshow(mark_boundaries(img, segments_slic))
ax[1].set_title("SLIC")
ax[2].imshow(mark_boundaries(img, segments_seeds))
ax[2].set_title("SEEDS")
ax[3].imshow(mark_boundaries(img, segments_ncuts))
ax[3].set_title("Ncuts on SLIC")

plt.show()

pdb.set_trace()

