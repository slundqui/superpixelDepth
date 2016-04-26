from dataObj.kitti import kittiObj
from tf.depthInference import unaryDepthInference

imageList = "/home/sheng/mountData/datasets/kitti/list/tf/trainImg.txt"
depthList = "/home/sheng/mountData/datasets/kitti/list/tf/trainDepth.txt"

saveFile = "/home/sheng/mountData/unaryDepthInference/model0_save.ckpt"

#Get object from which tensorflow will pull data from
dataObj = kittiObj(imageList, depthList)
tfObj = unaryDepthInference(dataObj)
print "Done init"
tfObj.trainModel(1000, saveFile)
print "Done run"
tfObj.plotLoss()
pdb.set_trace()






