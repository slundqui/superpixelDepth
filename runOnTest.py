import matplotlib
matplotlib.use('Agg')
from dataObj.kitti import kittiObj
from tf.depthInference import unaryDepthInference
from plot.plot import plotLoss, plotDepth, plotImg
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os

imageList = "/home/sheng/mountData/datasets/kitti/list/tf/testImg.txt"
depthList = "/home/sheng/mountData/datasets/kitti/list/tf/testDepth.txt"

trainImageList = "/home/sheng/mountData/datasets/kitti/list/tf/trainImg.txt"
trainDepthList = "/home/sheng/mountData/datasets/kitti/list/tf/trainDepth.txt"

outDir = "/home/sheng/mountData/unaryDepthInference/"
runDir = outDir + "/testRun/"
plotDir = runDir + "plots/"

if not os.path.exists(runDir):
   os.makedirs(runDir)

if not os.path.exists(plotDir):
   os.makedirs(plotDir)

load = True
loadFile = outDir + "/saved/saved.ckpt"

#Get object from which tensorflow will pull data from
testDataObj = kittiObj(imageList, depthList)

#Allocate obj to calc mean/std
trainDataObj = kittiObj(trainImageList, trainDepthList)

#Set mean/std on test set
testDataObj.setMeanVar(trainDataObj.mean, trainDataObj.std)

vggFile = "/home/sheng/mountData/pretrain/imagenet-vgg-f.mat"
#Allocate tf obj with test data
tfObj = unaryDepthInference(testDataObj, vggFile)

#Load weights
if(load):
   tfObj.loadModel(loadFile)
else:
   tfObj.initSess()

#Summary dir
tfObj.writeSummary(runDir + "/test")

print "Done init"
numImages = testDataObj.numImages

allGT = None
allEst = None

for i in range(numImages):
    print i
    #Evaluate current frame
    (evalData, gtData) = testDataObj.allSegments()
    estData = tfObj.evalModelBatch(32, evalData)
    if(allGT == None):
        allGT = gtData
    else:
        allGT = np.concatenate((allGT, gtData), axis=0)
    if(allEst == None):
        allEst = estData
    else:
        allEst = np.concatenate((allEst, estData), axis=0)

    plotDepth(testDataObj.currImage, testDataObj.currSegments, testDataObj.segLabels, gtData, estData, plotDir + "/gtVsEst_" + str(i) + ".png")

    gtOrig = np.exp(gtData) * trainDataObj.std
    estOrig = np.exp(estData) * trainDataObj.std
    plotDepth(testDataObj.currImage, testDataObj.currSegments, testDataObj.segLabels, gtOrig, estOrig, plotDir + "/gtVsEstOrig_" + str(i) + ".png")
    #Get next image
    testDataObj.nextImage()

print "Done run"
tfObj.closeSess()

#Undo normalization
est = np.exp(allEst) * trainDataObj.std
gt = np.exp(allGT) * trainDataObj.std

#Find average relative error
relE = np.mean(np.abs(gt - est)/gt)
print "rel: ", relE
#log10 error
logE = np.mean(np.abs(np.log10(gt) - np.log10(est)))
print "log10: ", logE
#rms
rmsE = np.sqrt(np.mean(np.power(gt - est,2)))
print "rms: ", rmsE







