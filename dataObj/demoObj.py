import scipy.io as spio
from scipy.ndimage import imread
import numpy as np
from segment.segment import segmentDepth, calcSegments, fillSegments
#import matplotlib.pyplot as plt
import pdb
from random import shuffle

class demoObj:
    inputShape = (224, 224, 3)

    def __init__(self, imgFile):
        self.loadImage(imgFile)

    #Function to return new image and depth file
    #TODO generate random ranking and randomize images
    def loadImage(self, imgFile):
        print "Updating Image to ", imgFile
        #Read data, with normalization to be -1 to 1
        self.currImage = imread(imgFile).astype(np.float32)/256
        #We don't use mean, std at all, just coords, which is unaffected by first parameter
        #Segment image
        self.currSegments = calcSegments(self.currImage)
        (segMean, segStd, segCoords, segLabels) = segmentDepth(self.currImage[:, :, 1], self.currSegments)
        self.segCoords = segCoords
        self.segLabels = segLabels


    #Crop image based on segments, give a label, and return both
    def nextSegment(self, updateImage=True):
        (ny, nx, nf) = self.currImage.shape
        segIdx = self.segmentIdx
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
        if(self.segmentIdx >= len(self.segLabels)):
            self.segmentIdx = 0
        return padImg

    #Get all segments of current image
    def allSegments(self):
       #Reset segmentIdx to 0
       self.segmentIdx = 0
       numSegments = len(self.segLabels)
       outVals = np.zeros((numSegments, self.inputShape[0], self.inputShape[1], self.inputShape[2]))

       for i in range(numSegments):
          data = self.nextSegment(updateImage=False)
          outVals[i, :, :, :] = data
       assert(self.segmentIdx == 0)
       #Restore segmentIdx
       return outVals

