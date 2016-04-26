from dataObj.kitti import kittiObj
from tf.depthInference import unaryDepthInference

imageList = "/home/sheng/mountData/datasets/kitti/list/image_2_benchmark_train_single.txt"
depthList = "/home/sheng/mountData/datasets/kitti/list/benchmark_depth_disp_noc.txt"

saveFile = "/home/sheng/mountData/unaryDepthInference/model0_save.ckpt"

#Get object from which tensorflow will pull data from
dataObj = kittiObj(imageList, depthList)
tfObj = unaryDepthInference(dataObj)
print "Done init"
tfObj.trainModel(1000, saveFile)
print "Done run"
tfObj.saveModel(saveFile)
tfObj.plotLoss()
pdb.set_trace()






