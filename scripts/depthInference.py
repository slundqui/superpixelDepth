import os, sys

import pdb
import numpy as np
import tensorflow as tf

from segment import pvSparseToDense, cropTopLeft, calcSegments, makeDataset

from skimage.util import img_as_float

from segment import makeDataset, readData

pvpFilename = "/home/sheng/mountData/benchmark/icaweights_binoc_LCA_fine/a12_V1.pvp"
imageList = "/home/sheng/mountData/datasets/kitti/list/image_2_benchmark_train_single.txt"
depthList = "/home/sheng/mountData/datasets/kitti/list/benchmark_depth_disp_noc.txt"

saveFile = "output/depthInfModel.ckpt"

minibatchSize = 4
weightInitStd = .1
biasInitConst = .1
runtime = 100000
progress = 100

pvpCropShape = (7, 7)
numInNodes = 7*7*512
numHidden1Nodes = 4092
numHidden2Nodes = 128
numHidden3Nodes = 16
numOutNodes = 1

scaleFactor = .25

imageIdx = 0 #Starting with 1 to skip first image
instanceIdx = 0

trainRange = range(1,100)
testRange = range(101,194)

currInput = np.array([])
currGT = np.array([])

#For initializing weights and biases
def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=weightInitStd)
   return tf.Variable(initial)

def bias_variable(shape):
   initial = tf.constant(.1, shape=shape)
   return tf.Variable(initial)

def node_variable(shape):
   return tf.placeholder("float", shape=shape)

def generate_feed_dict(minibatchSize, allImages, allDepth, pvpData, inImageRange):
   global imageIdx
   global instanceIdx
   global currInput
   global currGT

   outData = np.zeros((minibatchSize, numInNodes))
   outGT = np.zeros((minibatchSize, numOutNodes))

   for b in range(minibatchSize):
      numInstances = len(currInput)
      #If out of examples from image, grab new image
      if(instanceIdx >= numInstances):
         #Loop if out of images
         if(imageIdx >= len(inImageRange)):
            imageIdx = 0
         (currInput, currGT, drop, drop) = makeDataset(allImages, allDepth, pvpData, scaleFactor, pvpCropShape, 1, inImageRange[imageIdx])
         #Shuffle currInput and currGT
         randIdxs = np.random.permutation(len(currInput))
         currInput = currInput[randIdxs]
         currGT = currGT[randIdxs]

         imageIdx += 1
         instanceIdx = 0
      outData[b, :] = currInput[instanceIdx, :]
      outGT[b, :] = currGT[instanceIdx]
      instanceIdx += 1

   return (outData, outGT)



sess = tf.InteractiveSession()

#Get convolution variables as placeholders
inNode = node_variable([minibatchSize, numInNodes])
gtNode = node_variable([minibatchSize, numOutNodes])

#Model variables for an MLP
W1 = weight_variable([numInNodes, numHidden1Nodes])
B1 = bias_variable([numHidden1Nodes])
W2 = weight_variable([numHidden1Nodes, numHidden2Nodes])
B2 = bias_variable([numHidden2Nodes])
W3 = weight_variable([numHidden2Nodes, numHidden3Nodes])
B3 = bias_variable([numHidden3Nodes])
W4 = weight_variable([numHidden3Nodes, numOutNodes])
B4 = bias_variable([numOutNodes])

#Feedforward pass
hidden1Node = tf.nn.relu(tf.matmul(inNode, W1) + B1)
hidden2Node = tf.nn.relu(tf.matmul(hidden1Node, W2) + B2)
hidden3Node = tf.nn.sigmoid(tf.matmul(hidden2Node, W3) + B3)
outNode = tf.matmul(hidden3Node, W4) + B4

#Least squares loss
loss = tf.reduce_sum(tf.square(gtNode - outNode))

#Optimize with Adam algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

#Initialize
init_op = tf.initialize_all_variables()

#Get saver
saver = tf.train.Saver()

#Read all data
(allImages, allDepth, pvpData) = readData(imageList, depthList, pvpFilename)

#Buffer to store last progress number of loss values
lossVals = np.zeros((progress))

sess.run(init_op)

#Train
for i in range(runtime):
    (inputVals, gtSingleData) = generate_feed_dict(minibatchSize, allImages, allDepth, pvpData, trainRange)
    if i%progress == 0:
        lossVals[i%progress] = loss.eval(feed_dict={inNode: inputVals, gtNode: gtSingleData})
        #train_accuracy = accuracy.eval(feed_dict={inNode: inputVals, gtNode: gtSingleData})
        print "step: %d; curr training loss: %f; last %d loss: %f"%(i, lossVals[i%progress], progress, np.mean(lossVals))
    train_step.run(feed_dict={inNode: inputVals, gtNode: gtSingleData})

#Save model
save_path = saver.save(sess, saveFile)
print("Model saved in file: %s" % save_path)


#TO load:
#saver.restore(sess, "/tmp/model.ckpt")
#print("Model restored.")

pdb.set_trace()
