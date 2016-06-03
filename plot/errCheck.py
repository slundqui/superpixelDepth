import numpy as np

estFile = "/home/sheng/mountData/unaryDepthInference/testRunPreTrain/est.npy"
gtFile = "/home/sheng/mountData/unaryDepthInference/testRunPreTrain/gt.npy"

trainMean = 39.0076
trainStd = 19.1511

est = np.load(estFile)
gt = np.load(gtFile)

#Undo normalization
est = np.exp(est) * trainStd
gt = np.exp(gt) * trainStd

#Find average relative error
relE = np.mean(np.abs(gt- est)/gt)
print "rel: ", relE
#rms
rmsE = np.sqrt(np.mean(np.power(gt- est,2)))
print "rms: ", rmsE
#log10 error
logE = np.mean(np.abs(np.log10(gt) - np.log10(est)))
print "log: ", logE






