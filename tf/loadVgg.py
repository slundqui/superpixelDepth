import numpy as np
from scipy.io import loadmat
import pdb
import matplotlib.pyplot as plt

def convertW(inW):
    #inW is in the shape of [W, H, inF, outF]
    #We need to translate to [H, W, inF, outF]
    return np.transpose(inW, (1, 0, 2, 3))

def convertB(inB):
    return inB[:, 0]

def loadWeights(inFile):
    m = loadmat(inFile)

    outdict = {}

    outdict["conv1_w"] = convertW(m["layers"][0, 0][0, 0][2][0, 0])
    outdict["conv1_b"] = convertB(m["layers"][0, 0][0, 0][2][0, 1])
    outdict["conv2_w"] = convertW(m["layers"][0, 4][0, 0][2][0, 0])
    outdict["conv2_b"] = convertB(m["layers"][0, 4][0, 0][2][0, 1])
    outdict["conv3_w"] = convertW(m["layers"][0, 8][0, 0][2][0, 0])
    outdict["conv3_b"] = convertB(m["layers"][0, 8][0, 0][2][0, 1])
    outdict["conv4_w"] = convertW(m["layers"][0, 10][0, 0][2][0, 0])
    outdict["conv4_b"] = convertB(m["layers"][0, 10][0, 0][2][0, 1])
    outdict["conv5_w"] = convertW(m["layers"][0, 12][0, 0][2][0, 0])
    outdict["conv5_b"] = convertB(m["layers"][0, 12][0, 0][2][0, 1])

    return outdict


if __name__ == "__main__":
    inputFile = "/home/sheng/mountData/pretrain/imagenet-vgg-f.mat"
    outdict = loadWeights(inputFile)
    outmat = np.zeros((11*8, 11*8, 3))
    for i in range(64):
        x = i % 8
        y = i / 8
        xpos = x*11
        ypos = y*11
        outmat[ypos:ypos+11, xpos:xpos+11, :] = outdict["conv1_w"][:, :, :, i]

    plt.imshow(outmat)
    plt.show()

    pdb.set_trace()
