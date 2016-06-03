import matplotlib
matplotlib.use('Agg')
from dataObj.demoObj import demoObj
from tf.depthInference import unaryDepthInference
from plot.plot import plotSegments, plotEval
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
import os

#Get command line arguments
argv = sys.argv
if(len(argv) != 2):
    print("Usage: python demo.py [filename]")
    exit(-1)
inputFilename = argv[1]

outDir = "/home/sheng/mountData/unaryDepthInference/"
runDir = outDir + "/demo/"
plotDir = runDir + "plots/"

if not os.path.exists(runDir):
   os.makedirs(runDir)

if not os.path.exists(plotDir):
   os.makedirs(plotDir)

load = True
#TODO change loadfile
loadFile = outDir + "/saved/saved.ckpt"

#Allocate obj to calc mean/std
imgObj = demoObj(inputFilename)
#plotSegments(imgObj.currImage, imgObj.currSegments)

#Allocate tf obj with test data
tfObj = unaryDepthInference(imgObj, None)

tfObj.loadModel(loadFile)
evalData = imgObj.allSegments()
estData = tfObj.evalModelBatch(32, evalData)
plotEval(imgObj.currImage, imgObj.currSegments, imgObj.segLabels, estData, inputFilename + ".depth.png")



#Evaluate current frame

##Load weights
#if(load):
#else:
#   tfObj.initSess()
#
##Summary dir
#tfObj.writeSummary(runDir + "/test")
#
#print "Done init"
#numImages = testDataObj.numImages
#
#allGT = None
#allEst = None
#
#for i in range(numImages):
#    print i
#    #Evaluate current frame
#    (evalData, gtData) = testDataObj.allSegments()
#    estData = tfObj.evalModelBatch(32, evalData)
#    if(allGT == None):
#        allGT = gtData
#    else:
#        allGT = np.concatenate((allGT, gtData), axis=0)
#    if(allEst == None):
#        allEst = estData
#    else:
#        allEst = np.concatenate((allEst, estData), axis=0)
#
#    plotDepth(testDataObj.currImage, testDataObj.currSegments, testDataObj.segLabels, gtData, estData, plotDir + "/gtVsEst_" + str(i) + ".png")
#    #Get next image
#    testDataObj.nextImage()
#
#print "Done run"
#tfObj.closeSess()
#
##Undo normalization
#est = np.exp(allEst) * trainDataObj.std
#gt = np.exp(allGt) * trainDataObj.std
#
##Find average relative error
#relE = np.mean(np.abs(gt - est)/gt)
#print "rel: ", relE
##log10 error
#logE = np.mean(np.abs(np.log10(gt) - np.log10(est)))
#print "log10: ", logE
##rms
#rmsE = np.sqrt(np.mean(np.power(gt - est,2)))
#print "rms: ", rmsE
#
#
#
#
#
#
#
