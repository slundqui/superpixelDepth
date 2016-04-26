import scipy.io as spio
from skimage import io as imgio
import numpy as np
from segment import segmentDepth, calcSegments
import pdb

def readList(filename):
    f = open(filename, 'r')
    allLines = f.readlines()
    f.close()
    #Remove newlines from all lines
    return [line[:-1] for line in allLines]

class make3dObj:
    imgIdx = 0
    segmentIdx = 0

    def __init__(self, imgList, depthList):
        self.imgFiles = readList(imgList)
        self.depthFiles = readList(depthList)
        self.nextImage()

    #Function to return new image and depth file
    #TODO generate random ranking and randomize images
    def nextImage(self):
        imgFile = self.imgFiles[self.imgIdx]
        depthFile = self.depthFiles[self.imgIdx]

        #Read image
        self.currImage = imgio.imread(imgFile)

        #Read depth
        depthMat = spio.loadmat(depthFile)
        self.currDepth = depthMat["Position3DGrid"]
        self.imgIdx = (self.imgIdx + 1) % len(self.imgFiles)
        pdb.set_trace()

    def getData(self, numExample):
        for i in range(numExample):
            pass






if __name__ == "__main__":
    baseDir = "/home/sheng/mountData/datasets/make3d/"
    imgList = baseDir + "trainImg.txt"
    depthList = baseDir + "trainDepth.txt"
    obj = make3dObj(imgList, depthList)
    obj.nextImage()
