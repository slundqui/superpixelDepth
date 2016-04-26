import scipy.io as spio
from skimage import io as imgio
import numpy as np
from segment.segment import segmentDepth, calcSegments
import matplotlib.pyplot as plt
import pdb


def readList(filename):
    f = open(filename, 'r')
    allLines = f.readlines()
    f.close()
    #Remove newlines from all lines
    return [line[:-1] for line in allLines]

class kittiObj:
    imgIdx = 0
    segmentIdx = 0
    inputShape = (224, 224, 3)

    def __init__(self, imgList, depthList):
        self.imgFiles = readList(imgList)
        self.depthFiles = readList(depthList)
        assert(len(self.imgFiles) == len(self.depthFiles))
        self.nextImage()
        self.nextSegment()

    #Function to return new image and depth file
    #TODO generate random ranking and randomize images
    def nextImage(self):
        imgFile = self.imgFiles[self.imgIdx]
        depthFile = self.depthFiles[self.imgIdx]
        #Read data
        self.currImage = imgio.imread(imgFile).astype(np.float32)/255
        #Here, each pixel is uint16. We change it to uint8 by dividing by 256.
        #Additionally, most pixels are with range 0 to 128, so we divide by 128
        #to map from 0 to 1
        self.currDepth = imgio.imread(depthFile).astype(np.float32)/(256 * 128)
        #Update imgIdx
        self.imgIdx = (self.imgIdx + 1) % len(self.imgFiles)

        #Segment image
        self.currSegments = calcSegments(self.currImage)
        (segDepth, segMean, segStd, segCoords, segLabels) = segmentDepth(self.currDepth, self.currSegments)
        #We only need segMean (gt) and segCoords (for input parsing)
        self.segVals = segMean
        self.segCoords = segCoords
        assert(len(self.segVals) == len(self.segCoords))

    #Crop image based on segments, give a label, and return both
    def nextSegment(self):
        (ny, nx, nf) = self.currImage.shape
        outGt = self.segVals[self.segmentIdx]
        #Find centroid
        coords = self.segCoords[self.segmentIdx]
        centerY = coords[0] + (coords[2]/2)
        centerX = coords[1] + (coords[3]/2)
        topIdx = centerY - (self.inputShape[0]/2)
        botIdx = centerY + (self.inputShape[0]/2)
        leftIdx = centerX - (self.inputShape[1]/2)
        rightIdx = centerX + (self.inputShape[1]/2)
        #We pad edges with 0
        padTop = 0
        padBot = 0
        padLeft = 0
        padRight = 0
        if topIdx < 0:
            padTop = -topIdx
            topIdx = 0
        if botIdx >= ny:
            padBot = botIdx - ny + 1
            botIdx = ny - 1
        if leftIdx < 0:
            padLeft = -leftIdx
            leftIdx = 0
        if rightIdx >= nx:
            padRight = rightIdx - nx + 1
            rightIdx = nx - 1
        cropImg = self.currImage[topIdx:botIdx, leftIdx:rightIdx, :]
        padImg = np.pad(cropImg, ((padTop, padBot), (padLeft, padRight), (0, 0)), 'constant')

        #Update segmentIdx, and check if we need new image
        self.segmentIdx += 1
        if(self.segmentIdx >= len(self.segVals)):
            self.nextImage()
            self.segmentIdx = 0

        return (padImg, outGt)

    def getData(self, numExample):
        outData = np.zeros((numExample, cropShape[0], cropShape[1], 3))
        for i in range(numExample):
            outData[i, :, :, :] = self.nextSegment()
        return outData

#if __name__ == "__main__":
#    imageList = "/home/sheng/mountData/datasets/kitti/list/image_2_benchmark_train_single.txt"
#    depthList = "/home/sheng/mountData/datasets/kitti/list/benchmark_depth_disp_noc.txt"
#    obj = kittiObj(imageList, depthList)
#    #obj.getData(1)[0, :, :, :]



