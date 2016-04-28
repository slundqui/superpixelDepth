from dataObj.kitti import kittiObj
from tf.depthInference import unaryDepthInference
from plot.plot import plotLoss, plotDepth, plotImg
import matplotlib.pyplot as plt
import pdb
import os

imageList = "/home/sheng/mountData/datasets/kitti/list/tf/trainImg.txt"
depthList = "/home/sheng/mountData/datasets/kitti/list/tf/trainDepth.txt"

outDir = "/home/sheng/mountData/unaryDepthInference/"
runDir = outDir + "/run0/"
plotDir = runDir + "plots/"

if not os.path.exists(runDir):
   os.makedirs(runDir)

if not os.path.exists(plotDir):
   os.makedirs(plotDir)

load = False
loadFile = outDir + "/run0/model0_save.ckpt"

#Get object from which tensorflow will pull data from
dataObj = kittiObj(imageList, depthList)

#Get all segments
#(drop, gt) = dataObj.allSegments()
#plotImg(dataObj.currSegments, dataObj.segLabels, gt)
#plt.imshow(dataObj.currImage)
#plt.show()
#plt.imshow(dataObj.currDepth)
#plt.show()
#
#pdb.set_trace()

tfObj = unaryDepthInference(dataObj)

tfObj.writeSummary(runDir + "/train")

if(load):
   tfObj.loadModel(loadFile)
else:
   tfObj.initSess()

print "Done init"

for i in range(100):
   saveFile = runDir + "/model" + str(i) + ".ckpt"
   tfObj.trainModel(100, saveFile)
   #Evaluate current frame
   (evalData, gtData) = dataObj.allSegments()
   estData = tfObj.evalModel(evalData)
   plotDepth(dataObj.currSegments, dataObj.segLabels, gtData, estData, plotDir + "/gtVsEst_" + str(i) + ".png")

print "Done run"
plotLoss(tfObl.lossVals, plotDir + "/loss.png")

tfObj.closeSess()






