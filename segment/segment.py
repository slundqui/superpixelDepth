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

demo = True

#Here, we return a 5 tuple
#segMean: length numSegments, the mean of values per segment
#segStd: length numSegments, the std of values per segment
#segCoords: length of numSegments, describes each segment location in (nyTop, nxLeft, height, width)
#segLabels: length of numSegments, corresponds to the label of each segment in input segments
def segmentDepth(depth, segments):
    segMean = []
    segStd = []
    segCoords = []
    segLabels = []
    #Segment depth based on segmentations
    for label in np.unique(segments):
        #Find average depth of superpixel, not counting 0s
        spIdxs = np.nonzero(segments == label)
        spValidIdxs = np.nonzero(depth[spIdxs] != 0)
        #If spValidIdxs is empty, set as 0
        if spValidIdxs[0].size == 0:
            val = 0
        else:
            listVals = depth[spIdxs]
            val = np.mean(listVals[spValidIdxs])
            segMean.append(val)
            segStd.append(np.std(listVals[spValidIdxs]))
            #Stored as (nyTop, nxLeft, height, width)
            nyTop = np.min(spIdxs[0])
            nxLeft = np.min(spIdxs[1])
            nyBot = np.max(spIdxs[0])
            nxRight = np.max(spIdxs[1])
            segCoords.append((nyTop, nxLeft, nyBot-nyTop, nxRight-nxLeft))
            segLabels.append(label)
    return (segMean, segStd, segCoords, segLabels)

#nsegments Number of segments
#compactness Balances color and space proximity
#Higher gives more uniform segments
def calcSegments(image, nsegments = 1000, compactness = 10):
    #Width of gaussian smoothing kernel for preprocessing
    sigma = 1
    #SLIC segmentation
    segments= slic(image, n_segments=nsegments, compactness=compactness, sigma=sigma)
    return segments

def fillSegments(segments, vals, labels):
   assert(len(vals) == len(labels))
   (ny, nx) = segments.shape
   outFill = np.zeros((ny, nx))
   for (val, label) in zip(vals, labels):
      spIdxs = np.nonzero(segments == label)
      outFill[spIdxs] = val
   return outFill


#def pvSparseToDense(pvData, idx):
#    hdr = pvData.header
#    values = np.array(pvData[idx].values)
#
#    outMat = np.zeros(hdr["ny"]* hdr["nx"]* hdr["nf"])
#    outMat[values[:, 0].astype(int)] = values[:, 1]
#    outMat = outMat.reshape((hdr["ny"], hdr["nx"], hdr["nf"]), order='C')
#
#    return outMat

#def makeSingleDataset(pvp, segCoords, cropShape, scaleFactor):
#    assert(cropShape[0] % 2 == 1)
#    assert(cropShape[1] % 2 == 1)
#    centerOffsetY = np.floor(cropShape[0]/2)
#    centerOffsetX = np.floor(cropShape[1]/2)
#
#    numInstances = len(segCoords)
#    (ny, nx, nf) = pvp.shape
#    numFeatures = cropShape[0] * cropShape[1] * nf
#    outDataset = np.zeros((numInstances, numFeatures))
#
#    for i in range(numInstances):
#        coords = segCoords[i]
#        #Find center x and  y
#        centery = coords[0] + (coords[2]/2)
#        centerx = coords[1] + (coords[3]/2)
#        #Scale centers
#        scaleCenterY = np.floor(centery * scaleFactor)
#        scaleCenterX = np.floor(centerx * scaleFactor)
#
#        #Offsets into the feature mat
#        fyBegin= 0
#        fyEnd= cropShape[0]
#        fxBegin= 0
#        fxEnd= cropShape[1]
#        #Offsets into the pvp mat
#        pyBegin = scaleCenterY - centerOffsetY
#        pyEnd = scaleCenterY + centerOffsetY + 1
#        pxBegin = scaleCenterX - centerOffsetX
#        pxEnd = scaleCenterX + centerOffsetX + 1
##Check bounds and adjust indexes as necessary
#        if(pyBegin < 0):
#            fyBegin = -pyBegin
#            pyBegin = 0
#        if(pxBegin < 0):
#            fxBegin = -pxBegin
#            pxBegin = 0
#        if(pyEnd >= ny):
#            fyEnd = fyEnd - (pyEnd - ny) - 1
#            pyEnd = ny - 1
#        if(pxEnd >= nx):
#            fxEnd = fxEnd - (pxEnd - nx) - 1
#            pxEnd = nx - 1
#
#        #Out mat
#        featureMat = np.zeros((cropShape[0], cropShape[1], nf))
#
#        #Crop pvp file into featureMat
#        featureMat[fyBegin:fyEnd, fxBegin:fxEnd, :] = pvp[pyBegin:pyEnd, pxBegin:pxEnd, :]
#
#        #Flatten and store into outmat as a sparse matrix
#
#
#        outDataset[i, :] = featureMat.ravel()
#    return outDataset
#
##Crop image from the top left
#def cropTopLeft(inImg, targetY, targetX):
#    shape = np.shape(inImg)
#    #Only crop, no extend
#    assert(targetY <= shape[0] and targetX <= shape[1])
#    if(inImg.ndim == 2):
#        return inImg[shape[0]-targetY:, shape[1]-targetX:]
#    else:
#        return inImg[shape[0]-targetY:, shape[1]-targetX:, :]
#
#
#
#def readData(imageListFilename, depthListFilename, pvpFilename):
#    #Load pvp file
#    #pvpData = readpvpfile(pvpFilename)
#
#    #Load list
#    imageFile = open(imageListFilename, 'r')
#    depthFile = open(depthListFilename, 'r')
#    allImages = imageFile.readlines()
#    allDepth = depthFile.readlines()
#    imageFile.close()
#    depthFile.close()
#    return (allImages, allDepth)
#
#def makePvpSegments(imageListFilename, outPvpFilename):
#    imageFile = open(imageListFilename, 'r')
#    allImages = imageFile.readlines()
#    imageFile.close()
#    #Read first image to get shape of image
#    img = img_as_float(io.imread(allImages[0][:-1]))
#    (imgY, imgX, drop) = img.shape
#
#    #Open output pvpfile for writing
#    outMatFile = open(outPvpFilename, 'wb')
#    #Write header out to file
#    writeHeaderFile(outMatFile, (imgY, imgX, 1), len(allImages))
#
#    for (i, imFilename) in enumerate(allImages):
#        #Remove newline character
#        singleImgFilename = imFilename[:-1]
#        print singleImgFilename
#        img = img_as_float(io.imread(singleImgFilename))
#        segments = calcSegments(img)
#        #Add single dimension to the end, as we have 1 feature
#        segments = np.expand_dims(segments, 2)
#        #Write data out to file
#        writeData(outMatFile, segments, i)
#    #Close pvp file
#    outMatFile.close()
#
#
#def makeDataset(allImages, allDepth, pvpData, scaleFactor, pvpCropShape, numImages, offset):
#
#    global demo
#
#    numPvpFrames = pvpData.header["nbands"]
#    numLoop = np.min([numPvpFrames, len(allImages), len(allDepth)])
#
#    outData = None
#    outGt = None
#
#    imgIdx = 0;
#    #Loop through image lists
#    for (pvpIdx, (imfn, defn)) in enumerate(zip(allImages, allDepth)):
#        #Skip first index, as it's a repeat
#        if(pvpIdx == 0):
#            continue
#        if (pvpIdx < offset):
#            continue
#        if(pvpIdx >= numLoop or imgIdx >= numImages):
#            break
#
#        imageFilename = allImages[pvpIdx][:-1]
#        depthFilename = allDepth[pvpIdx][:-1]
#        print imageFilename
#
#        #Get pvp frame
#        singlePvp = pvSparseToDense(pvpData, pvpIdx)
#        (pvpy, pvpx, pvpnf) = singlePvp.shape
#
#        img = img_as_float(io.imread(imageFilename))
#        depth = imread(depthFilename)
#        depth = depth.astype(np.float64)/256
#
#        #Crop img and depth to match pvp data
#        img = cropTopLeft(img, pvpy/scaleFactor, pvpx/scaleFactor)
#        depth = cropTopLeft(depth, pvpy/scaleFactor, pvpx/scaleFactor)
#
#        segments = calcSegments(img)
#        (segDepth, segMeans, segStd, segCoords, segLabels) = segmentDepth(depth, segments)
#
#        if demo:
#            plotFig(img, segments, depth, segDepth)
#            pdb.set_trace()
#
#        #Get dataset from single image
#        singleData = makeSingleDataset(singlePvp, segCoords, pvpCropShape, scaleFactor)
#        singleGt = np.array(segMeans)/128
#
#        ##Sort into training or testing
#        #if(pvpIdx < trainTestSplit[0]):
#        if outData == None:
#            outData = singleData
#            outGt = singleGt
#        else:
#            np.concatenate((outData, singleData), 0)
#            np.concatenate((outGt, singleGt), 0)
#        imgIdx += 1
#        #else:
#        #    if testData == None:
#        #        testData = outData
#        #        testGt = outGt
#        #    else:
#        #        np.concatenate((testData, outData), 0)
#        #        np.concatenate((testGt, outGt), 0)
#
#    return (outData, outGt, segLabels, segments)



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
    ax[2].imshow(depth, cmap=colormap, vmin=.0001)
    ax[2].set_title("Orig depth")
    axx = ax[3].imshow(segDepth, cmap=colormap, vmin=.0001)
    ax[3].set_title("Seg depth")

    f.subplots_adjust(right=.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(axx, cax=cbar_ax)

    plt.figure()
    plt.hist(depth[np.nonzero(depth != 0)].flatten())
    #ax[4].imshow(img)
    #ax[4].imshow(depth, alpha=.5, vmin=.0001)
    #ax[4].set_title("Orig depth")
    #ax[5].imshow(img)
    #ax[5].imshow(segDepth, alpha=.5, vmin=.0001)
    #ax[5].set_title("Seg depth")

    #f.colorbar(axx)

    plt.show()

if __name__ == "__main__":
    imageList = "/home/sheng/mountData/datasets/kitti/list/tf/trainImg.txt"
    depthList = "/home/sheng/mountData/datasets/kitti/list/tf/trainDepth.txt"

    #pvpOutFilename = "kittiSeg.pvp"
    #makePvpSegments(imageList, pvpOutFilename)

    scaleFactor = .25
    #100 for training, 94 for testing
    numImages = 1
    offset = 20

    ##Patch to take around centroid of superpixel
    #pvpCropShape = (33, 33)

    #(allImages, allDepth) = readData(imageList, depthList)

    #(data, gt) = makeDataset(allImages, allDepth, pvpData, scaleFactor, pvpCropShape, numImages, offset)


    pdb.set_trace()




