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

def segmentDepth(depth, segments):
    segDepth = np.zeros(depth.shape)
    segVals = []
    segCoords = []
    segLabels = []
    #Segment depth based on segmentations
    for label in np.unique(segments):
        #Find average depth of superpixel, not counting 0s
        spIdxs = np.nonzero(segments == label)
        meanIdxs = np.nonzero(depth[spIdxs] != 0)
        #If meanIdxs is empty, set as 0
        if meanIdxs[0].size == 0:
            val = 0
        else:
            listVals = depth[spIdxs]
            val = np.mean(listVals[meanIdxs])
            segVals.append(val)
            #Stored as (nyTop, nxLeft, height, width)
            nyTop = np.min(spIdxs[0])
            nxLeft = np.min(spIdxs[1])
            nyBot = np.max(spIdxs[0])
            nxRight = np.max(spIdxs[1])
            segCoords.append((nyTop, nxLeft, nyBot-nyTop, nxRight-nxLeft))
            segLabels.append(label)
        #Set all values to val
        segDepth[spIdxs] = val
    return (segDepth, segVals, segCoords, segLabels)

#nsegments Number of segments
#compactness Balances color and space proximity
#Higher gives more uniform segments
def calcSegments(image, nsegments = 2000, compactness = 20):
    #Width of gaussian smoothing kernel for preprocessing
    sigma = 1
    #SLIC segmentation
    segments= slic(image, n_segments=nsegments, compactness=compactness, sigma=sigma)
    return segments

def pvSparseToDense(pvData, idx):
    hdr = pvData.header
    values = np.array(pvData[idx].values)

    outMat = np.zeros(hdr["ny"]* hdr["nx"]* hdr["nf"])
    outMat[values[:, 0].astype(int)] = values[:, 1]
    outMat = outMat.reshape((hdr["ny"], hdr["nx"], hdr["nf"]), order='C')

    return outMat

def makeSingleDataset(pvp, segCoords, cropShape, scaleFactor):
    assert(cropShape[0] % 2 == 1)
    assert(cropShape[1] % 2 == 1)
    centerOffsetY = np.floor(cropShape[0]/2)
    centerOffsetX = np.floor(cropShape[1]/2)

    numInstances = len(segCoords)
    (ny, nx, nf) = pvp.shape
    numFeatures = cropShape[0] * cropShape[1] * nf
    outDataset = np.zeros((numInstances, numFeatures))

    for i in range(numInstances):
        coords = segCoords[i]
        #Find center x and  y
        centery = coords[0] + (coords[2]/2)
        centerx = coords[1] + (coords[3]/2)
        #Scale centers
        scaleCenterY = np.floor(centery * scaleFactor)
        scaleCenterX = np.floor(centerx * scaleFactor)

        #Offsets into the feature mat
        fyBegin= 0
        fyEnd= cropShape[0]
        fxBegin= 0
        fxEnd= cropShape[1]
        #Offsets into the pvp mat
        pyBegin = scaleCenterY - centerOffsetY
        pyEnd = scaleCenterY + centerOffsetY + 1
        pxBegin = scaleCenterX - centerOffsetX
        pxEnd = scaleCenterX + centerOffsetX + 1
#Check bounds and adjust indexes as necessary
        if(pyBegin < 0):
            fyBegin = -pyBegin
            pyBegin = 0
        if(pxBegin < 0):
            fxBegin = -pxBegin
            pxBegin = 0
        if(pyEnd >= ny):
            fyEnd = fyEnd - (pyEnd - ny) - 1
            pyEnd = ny - 1
        if(pxEnd >= nx):
            fxEnd = fxEnd - (pxEnd - nx) - 1
            pxEnd = nx - 1

        #Out mat
        featureMat = np.zeros((cropShape[0], cropShape[1], nf))

        #Crop pvp file into featureMat
        featureMat[fyBegin:fyEnd, fxBegin:fxEnd, :] = pvp[pyBegin:pyEnd, pxBegin:pxEnd, :]

        #Flatten and store into outmat as a sparse matrix


        outDataset[i, :] = featureMat.ravel()
    return outDataset

#Crop image from the top left
def cropTopLeft(inImg, targetY, targetX):
    shape = np.shape(inImg)
    #Only crop, no extend
    assert(targetY <= shape[0] and targetX <= shape[1])
    if(inImg.ndim == 2):
        return inImg[shape[0]-targetY:, shape[1]-targetX:]
    else:
        return inImg[shape[0]-targetY:, shape[1]-targetX:, :]


def makeDataset(imageList, depthList, pvpFilename, scaleFactor, pvpCropShape, trainTestSplit):

    #Load pvp file
    pvpData = readpvpfile(pvpFilename)

    #Load list
    imageFile = open(imageList, 'r')
    depthFile = open(depthList, 'r')
    allImages = imageFile.readlines()
    allDepth = depthFile.readlines()
    imageFile.close()
    depthFile.close()

    numPvpFrames = pvpData.header["nbands"]
    numLoop = np.min([numPvpFrames, len(allImages), len(allDepth), np.sum(trainTestSplit)])

    outData = None
    outGt = None

    #Loop through image lists
    for (imfn, defn) in zip(allImages, allDepth):
        #Skip first index, as it's a repeat
        if(pvpIdx == 0):
            continue

        imageFilename = allImages[pvpIdx][:-1]
        depthFilename = allDepth[pvpIdx][:-1]
        print imageFilename

        #Get pvp frame
        singlePvp = pvSparseToDense(pvpData, pvpIdx)
        (pvpy, pvpx, pvpnf) = singlePvp.shape


        img = img_as_float(io.imread(imageFilename))
        depth = imread(depthFilename)
        depth = depth.astype(np.float64)/256

        #Crop img and depth to match pvp data
        img = cropTopLeft(img, pvpy/scaleFactor, pvpx/scaleFactor)
        depth = cropTopLeft(depth, pvpy/scaleFactor, pvpx/scaleFactor)

        segments = calcSegments(img)
        (segDepth, segVals, segCoords, segLabels) = segmentDepth(depth, segments)

        #Get dataset from single image
        outData = makeSingleDataset(singlePvp, segCoords, pvpCropShape, scaleFactor)
        outGt = segVals

        ##Sort into training or testing
        #if(pvpIdx < trainTestSplit[0]):
        if outData == None:
            outData = outData
            trainGt = outGt
        else:
            np.concatenate((trainData, outData), 0)
            np.concatenate((trainGt, outGt), 0)
        #else:
        #    if testData == None:
        #        testData = outData
        #        testGt = outGt
        #    else:
        #        np.concatenate((testData, outData), 0)
        #        np.concatenate((testGt, outGt), 0)

    return trainData, trainGt

def plotFig(img, segments, depth, segDepth):
    #Plotting
    f, ax = plt.subplots(4, 1, sharex=True)

    #Use jet colormap, where value of vmin gets set to black (for DNC regions)
    colormap = cm.get_cmap('jet')
    colormap.set_under('black')

    ax[0].imshow(img)
    ax[0].set_title("Orig")
    ax[1].imshow(mark_boundaries(img, segments))
    ax[1].set_title("SLIC")
    axx = ax[2].imshow(depth, cmap=colormap, vmin=.0001)
    ax[2].set_title("Orig depth")
    ax[3].imshow(segDepth, cmap=colormap, vmin=.0001)
    ax[3].set_title("Seg depth")
    #ax[4].imshow(img)
    #ax[4].imshow(depth, alpha=.5, vmin=.0001)
    #ax[4].set_title("Orig depth")
    #ax[5].imshow(img)
    #ax[5].imshow(segDepth, alpha=.5, vmin=.0001)
    #ax[5].set_title("Seg depth")

    f.colorbar(axx)

    plt.show()

    pdb.set_trace()

if __name__ == "__main__":
    pvpFilename = "/home/sheng/mountData/benchmark/icaweights_binoc_LCA_fine/a12_V1.pvp"
    imageList = "/home/sheng/mountData/datasets/kitti/list/image_2_benchmark_train_single.txt"
    depthList = "/home/sheng/mountData/datasets/kitti/list/benchmark_depth_disp_noc.txt"

    scaleFactor = .25
    #100 for training, 94 for testing
    trainTestSplit = [100, 94]

    #Patch to take around centroid of superpixel
    pvpCropShape = (33, 33)

    (trainData, trainGt, testData, testGt) = makeDataset(imageList, depthList, pvpFilename, scaleFactor, pvpCropShape, trainTestSplit)


    pdb.set_trace()




