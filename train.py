import matplotlib
matplotlib.use('Agg')
from dataObj.kitti import kittiObj
from tf.depthInference import unaryDepthInference
from plot.plot import plotLoss, plotDepth, plotImg
import numpy as np
import pdb
import os

vggFile = "/home/sheng/mountData/pretrain/imagenet-vgg-f.mat"

trainImageList = "/home/sheng/mountData/datasets/kitti/list/tf/trainImg.txt"
trainDepthList = "/home/sheng/mountData/datasets/kitti/list/tf/trainDepth.txt"

testImageList = "/home/sheng/mountData/datasets/kitti/list/tf/testImg.txt"
testDepthList = "/home/sheng/mountData/datasets/kitti/list/tf/testDepth.txt"

outDir = "/home/sheng/mountData/unaryDepthInference/"
runDir = outDir + "/run0/"
plotDir = runDir + "plots/"

if not os.path.exists(runDir):
   os.makedirs(runDir)

if not os.path.exists(plotDir):
   os.makedirs(plotDir)

load = False
loadFile = outDir + "/saved/saved.ckpt"

#Get object from which tensorflow will pull data from
trainDataObj = kittiObj(trainImageList, trainDepthList)
testDataObj = kittiObj(testImageList, testDepthList)

testDataObj.setMeanVar(trainDataObj.mean, trainDataObj.std)

##Get all segments
#(drop, gt) = dataObj.allSegments()
#plotImg(dataObj.currSegments, dataObj.segLabels, gt)
#plotImg(dataObj.currSegments, dataObj.segLabels, np.log(gt))
#
#plt.hist(gt)
#plt.show()
#
#plt.hist(np.log(gt))
#plt.show()
#
#plt.set_trace()
#
#
#plt.imshow(dataObj.currImage)
#plt.show()
#plt.imshow(dataObj.currDepth)
#plt.show()
#
#pdb.set_trace()

tfObj = unaryDepthInference(trainDataObj, vggFile)

if(load):
   tfObj.loadModel(loadFile)
else:
   tfObj.initSess()

tfObj.writeSummary(runDir + "/tfout")

print "Done init"

#Pretrain
for i in range(100):
   saveFile = runDir + "/depth-model-pre"

   #Evaluate test frame, providing gt so that it writes to summary
   (evalData, gtData) = testDataObj.allSegments()
   estData = tfObj.evalModelBatch(32,evalData, gtData)
   print "Done test eval"
   plotDepth(testDataObj.currImage, testDataObj.currSegments, testDataObj.segLabels, gtData, estData, plotDir + "/gtVsEst_test_" + str(i) + ".png")
   print "Done test plot"

   #Evaluate train frame, and plot
   (evalData, gtData) = trainDataObj.allSegments()
   estData = tfObj.evalModelBatch(32, evalData)
   print "Done train eval"
   plotDepth(trainDataObj.currImage, trainDataObj.currSegments, trainDataObj.segLabels, gtData, estData, plotDir + "/gtVsEst_train_" + str(i) + ".png")
   print "Done train plot"

   #Train
   tfObj.trainModel(100, saveFile, pre=True)

#Train
for i in range(1000):
   saveFile = runDir + "/depth-model"

   #Evaluate test frame, providing gt so that it writes to summary
   (evalData, gtData) = testDataObj.allSegments()
   estData = tfObj.evalModelBatch(32,evalData, gtData)
   print "Done test eval"
   plotDepth(testDataObj.currImage, testDataObj.currSegments, testDataObj.segLabels, gtData, estData, plotDir + "/gtVsEst_test_" + str(i) + ".png")
   print "Done test plot"

   #Evaluate train frame, and plot
   (evalData, gtData) = trainDataObj.allSegments()
   estData = tfObj.evalModelBatch(32, evalData)
   print "Done train eval"
   plotDepth(trainDataObj.currImage, trainDataObj.currSegments, trainDataObj.segLabels, gtData, estData, plotDir + "/gtVsEst_train_" + str(i) + ".png")
   print "Done train plot"

   #Train
   tfObj.trainModel(100, saveFile, pre=False)

print "Done run"

tfObj.closeSess()






