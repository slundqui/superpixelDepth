import scipy.io as spio
from skimage import io as imgio
import numpy as np
from segment.segment import segmentDepth, calcSegments, fillSegments
import matplotlib.pyplot as plt
import pdb
from random import shuffle

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
        #self.nextSegment()

    #Function to return new image and depth file
    #TODO generate random ranking and randomize images
    def nextImage(self):
        print "Updating Image to ", self.imgFiles[self.imgIdx]
        imgFile = self.imgFiles[self.imgIdx]
        depthFile = self.depthFiles[self.imgIdx]
        #Read data, with normalization to be -1 to 1
        self.currImage = imgio.imread(imgFile).astype(np.float32)/256
        #Here, each pixel is uint16. We change it to uint8 by dividing by 256.
        #Additionally, most pixels are with range 0 to 128, so we divide by 128
        #to map from 0 to 1.
        self.currDepth = imgio.imread(depthFile).astype(np.float32)/(256 * 128)

        #Update imgIdx
        self.imgIdx = (self.imgIdx + 1) % len(self.imgFiles)

        #Segment image
        self.currSegments = calcSegments(self.currImage)
        (segMean, segStd, segCoords, segLabels) = segmentDepth(self.currDepth, self.currSegments)

        #Normalize ground truth here
        self.segVals = segMean
        self.segCoords = segCoords
        self.segLabels = segLabels

        assert(len(self.segVals) == len(self.segCoords))

        #Generate shuffled index based on how many segments
        self.shuffleIdx = range(len(self.segVals))
        shuffle(self.shuffleIdx)

    #Crop image based on segments, give a label, and return both
    def nextSegment(self, updateImage=True, shuffleIdx=True):
        (ny, nx, nf) = self.currImage.shape
        if(shuffleIdx):
           segIdx = self.shuffleIdx[self.segmentIdx]
        else:
           segIdx = self.segmentIdx
        outGt = self.segVals[segIdx]
        #Find centroid
        coords = self.segCoords[segIdx]
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
        #Normalize cropImg here
        #img range is 0 to 1, so change to -1 to 1
        cropImg = (cropImg * 2) - 1

        padImg = np.pad(cropImg, ((padTop, padBot), (padLeft, padRight), (0, 0)), 'constant')

        #Update segmentIdx, and check if we need new image
        self.segmentIdx += 1
        if(self.segmentIdx >= len(self.segVals)):
            #We put this flag for allSegments, as we don't want to
            #update the image when we call allSegments
            if(updateImage):
               self.nextImage()
            self.segmentIdx = 0

        return (padImg, outGt)

    #Get all segments of current image
    def allSegments(self):
       #Keep track of current index for nextSegment
       tmpSegmentIdx = self.segmentIdx
       #Reset segmentIdx to 0
       self.segmentIdx = 0
       numSegments = len(self.segVals)
       outVals = np.zeros((numSegments, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
       outGt = np.zeros((numSegments, 1))

       for i in range(numSegments):
          data = self.nextSegment(updateImage=False, shuffleIdx=False)
          outVals[i, :, :, :] = data[0]
          outGt[i, :] = data[1]

       assert(self.segmentIdx == 0)

       #Restore segmentIdx
       self.segmentIdx = tmpSegmentIdx
       return (outVals, outGt)

    def getData(self, numExample):
        outData = np.zeros((numExample, self.inputShape[0], self.inputShape[1], 3))
        outGt = np.zeros((numExample, 1))
        for i in range(numExample):
            data = self.nextSegment()
            outData[i, :, :, :] = data[0]
            outGt[i, :] = data[1]
        return (outData, outGt)





#if __name__ == "__main__":
#    imageList = "/home/sheng/mountData/datasets/kitti/list/image_2_benchmark_train_single.txt"
#    depthList = "/home/sheng/mountData/datasets/kitti/list/benchmark_depth_disp_noc.txt"
#    obj = kittiObj(imageList, depthList)
#    #obj.getData(1)[0, :, :, :]



