import h5py
from skimage import io as imgio
import numpy as np
#from segment import segmentDepth, calcSegments
import pdb
import matplotlib.pyplot as plt


def readList(filename):
    f = open(filename, 'r')
    allLines = f.readlines()
    f.close()
    #Remove newlines from all lines
    return [line[:-1] for line in allLines]

class nyuDepthObj:
    imgIdx = 0
    segmentIdx = 0

    def __init__(self, matFile):
        f = h5py.File(matFile)
        depth = f["depths"][0, :, :]
        n, bins, patches = plt.hist(np.log(depth.flatten()), 10)
        plt.show()
        pdb.set_trace()
        self.nextImage()

    #Function to return new image and depth file
    #TODO generate random ranking and randomize images
    def nextImage(self):
        #Read depth
        self.currDepth = depthMat["Position3DGrid"]
        self.imgIdx = (self.imgIdx + 1) % len(self.imgFiles)

        plt.imshow(currDepth)
        plt.colorbar()
        pdb.set_trace()

    def getData(self, numExample):
        for i in range(numExample):
            pass






if __name__ == "__main__":
    baseDir = "/home/sheng/mountData/datasets/nyuDepth/"
    matFile = baseDir + "nyu_depth_v2_labeled.mat"
    obj = nyuDepthObj(matFile)
    obj.nextImage()
