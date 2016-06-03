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
loadFile = "pretrain/saved.ckpt"

#Allocate obj to calc mean/std
imgObj = demoObj(inputFilename)
#plotSegments(imgObj.currImage, imgObj.currSegments)

#Allocate tf obj with test data
tfObj = unaryDepthInference(imgObj, None)

tfObj.loadModel(loadFile)
evalData = imgObj.allSegments()
estData = tfObj.evalModelBatch(32, evalData)
plotEval(imgObj.currImage, imgObj.currSegments, imgObj.segLabels, estData, inputFilename + ".depth.png")



