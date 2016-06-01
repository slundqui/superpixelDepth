import pdb
import numpy as np
import tensorflow as tf
from loadVgg import loadWeights
#import matplotlib.pyplot as plt

#For initializing weights and biases

def weight_variable_fromnp(inNp, inName):
    shape = inNp.shape
    return tf.Variable(inNp, name=inName)

def weight_variable_xavier(shape, inName, conv):
   #initial = tf.truncated_normal(shape, stddev=weightInitStd, name=inName)
   if conv:
       initial = tf.contrib.layers.xavier_initializer_conv2d()
   else:
       initial = tf.contrib.layers.xavier_initializer()
   return tf.get_variable(inName, shape, initializer=initial)

def weight_variable(shape, inName, inStd):
    initial = tf.truncated_normal_initializer(stddev=inStd)
    return tf.get_variable(inName, shape, initializer=initial)

def bias_variable(shape, inName, biasInitConst=.01):
   initial = tf.constant(biasInitConst, shape=shape, name=inName)
   return tf.Variable(initial)

def node_variable(shape, inName):
   return tf.placeholder("float", shape=shape, name=inName)

def conv2d(x, W, inName):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=inName)

def maxpool_2x2(x, inName):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=inName)

class unaryDepthInference:
    #Initialize tf parameters here
    progress = 1
    learningRate = 1e-4
    timestep = 0

    def __init__(self, dataObj, vggFile = None):
        self.dataObj = dataObj
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.buildModel(vggFile)

    def buildModel(self, inMatFilename = None):
        if(inMatFilename):
            npWeights = loadWeights(inMatFilename)

        #Put all conv layers on gpu
        with tf.device('gpu:0'):
            with tf.name_scope("inputOps"):
                inputShape = self.dataObj.inputShape
                #Get convolution variables as placeholders
                self.inputImage = node_variable([None, inputShape[0], inputShape[1], inputShape[2]], "inputImage")
                self.gt = node_variable([None, 1], "gt")
                #Model variables for convolutions

            with tf.name_scope("Conv1Ops"):
                if(inMatFilename):
                    self.W_conv1 = weight_variable_fromnp(npWeights["conv1_w"], "w_conv1")
                    self.B_conv1 = weight_variable_fromnp(npWeights["conv1_b"], "b_conv1")
                else:
                    #First conv layer is 11x11, 3 input channels into 64 output channels
                    self.W_conv1 = weight_variable_xavier([11, 11, 3, 64], "w_conv1", conv=True)
                    self.B_conv1 = bias_variable([64], "b_conv1")
                self.h_conv1 = tf.nn.relu(conv2d(self.inputImage, self.W_conv1, "conv1") + self.B_conv1)
                self.h_norm1 = tf.nn.local_response_normalization(self.h_conv1, name="LRN1")
                #relu is communative op, so do relu after pool for efficiency
                self.h_pool1 = maxpool_2x2(self.h_norm1, "pool1")

            with tf.name_scope("Conv2Ops"):
                #Second conv layer is 5x5 conv, into 256 output channels
                if(inMatFilename):
                    self.W_conv2 = weight_variable_fromnp(npWeights["conv2_w"], "w_conv2")
                    self.B_conv2 = weight_variable_fromnp(npWeights["conv2_b"], "b_conv2")
                else:
                    self.W_conv2 = weight_variable_xavier([5, 5, 64, 256], "w_conv2", conv=True)
                    self.B_conv2 = bias_variable([256], "b_conv2")
                self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2, "conv2") + self.B_conv2)
                self.h_norm2 = tf.nn.local_response_normalization(self.h_conv2, name="LRN2")
                self.h_pool2 = maxpool_2x2(self.h_norm2, "pool2")

            #Third layer is 3x3 conv into 256 output channels
            #No pooling
            with tf.name_scope("Conv3Ops"):
                #Second conv layer is 5x5 conv, into 256 output channels
                if(inMatFilename):
                    self.W_conv3 = weight_variable_fromnp(npWeights["conv3_w"], "w_conv3")
                    self.B_conv3 = weight_variable_fromnp(npWeights["conv3_b"], "b_conv3")
                else:
                    self.W_conv3 = weight_variable_xavier([3, 3, 256, 256], "w_conv3", conv=True)
                    self.B_conv3 = bias_variable([256], "b_conv3")
                self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3, "conv3") + self.B_conv3, name="relu3")

            #Fourth layer is 3x3 conv into 256 output channels
            #No pooling
            with tf.name_scope("Conv4Ops"):
                #Second conv layer is 5x5 conv, into 256 output channels
                if(inMatFilename):
                    self.W_conv4 = weight_variable_fromnp(npWeights["conv4_w"], "w_conv4")
                    self.B_conv4 = weight_variable_fromnp(npWeights["conv4_b"], "b_conv4")
                else:
                    self.W_conv4 = weight_variable_xavier([3, 3, 256, 256], "w_conv4", conv=True)
                    self.B_conv4 = bias_variable([256], "b_conv4")
                self.h_conv4 = tf.nn.relu(conv2d(self.h_conv3, self.W_conv4, "conv4") + self.B_conv4, name="relu4")

            #Fifth layer is 3x3 conv into 256 output channels
            #with pooling
            with tf.name_scope("Conv5Ops"):
                #Second conv layer is 5x5 conv, into 256 output channels
                if(inMatFilename):
                    self.W_conv5 = weight_variable_fromnp(npWeights["conv5_w"], "w_conv5")
                    self.B_conv5 = weight_variable_fromnp(npWeights["conv5_b"], "b_conv5")
                else:
                    self.W_conv5 = weight_variable_xavier([3, 3, 256, 256], "w_conv5", conv=True)
                    self.B_conv5 = bias_variable([256], "b_conv5")
                self.h_conv5 = tf.nn.relu(conv2d(self.h_conv4, self.W_conv5, "conv5") + self.B_conv5)
                self.h_norm5 = tf.nn.local_response_normalization(self.h_conv5, name="LRN5")
                self.h_pool5 = maxpool_2x2(self.h_norm5, "pool5")

            #6th layer (not in paper) is 3x3 conv into 256 output channels
            #with pooling
            with tf.name_scope("Conv6Ops"):
                self.W_conv6 = weight_variable_xavier([3, 3, 256, 256], "w_conv6", conv=True)
                self.B_conv6 = bias_variable([256], "b_conv6")
                self.h_conv6 = conv2d(self.h_pool5, self.W_conv6, "conv6") + self.B_conv6
                self.h_pool6 = tf.nn.relu(maxpool_2x2(self.h_conv6, "pool6"), name="relu6")

            self.keep_prob = tf.placeholder(tf.float32)

            #Next is 3 fully connected layers
            #We should have downsampled by 8 at this point
            #fc1 should have 4096 channels
            numInputs = (inputShape[0]/16) * (inputShape[1]/16) * 256
            with tf.name_scope("FC1"):
                self.W_fc1 = weight_variable([numInputs, 2048], "w_fc1", 1e-6)
                self.B_fc1 = bias_variable([2048], "b_fc1")
                h_pool6_flat = tf.reshape(self.h_pool6, [-1, numInputs], name="pool6_flat")
                self.h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat, self.W_fc1, name="fc1") + self.B_fc1, "fc1_relu")
                self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        #Put all opt layers on cpu
        with tf.device('/cpu:0'):

            #fc2 should have 128 channels
            with tf.name_scope("FC2"):
                self.W_fc2 = weight_variable_xavier([2048, 128], "w_fc2", conv=False)
                self.B_fc2 = bias_variable([128], "b_fc2")
                self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1_drop, self.W_fc2, name="fc2") + self.B_fc2, "fc2_relu")
                self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)

            #fc3 should have 16 channels
            #fc3 also uses a sigmoid function
            #We change it to tanh
            with tf.name_scope("FC3"):
                self.W_fc3 = weight_variable_xavier([128, 16], "w_fc3", conv=False)
                self.B_fc3 = bias_variable([16], "b_fc3")
                self.h_fc3 = tf.tanh(tf.matmul(self.h_fc2, self.W_fc3, name="fc3") + self.B_fc3, "fc3_relu")


            #Finally, fc4 condenses into 1 output value
            with tf.name_scope("FC4"):
                self.W_fc4 = weight_variable_xavier([16, 1], "w_fc4", conv=False)
                self.B_fc4 = bias_variable([1], "b_fc4")
                self.est = tf.matmul(self.h_fc3, self.W_fc4, name="est") + self.B_fc4

            with tf.name_scope("Loss"):
                #Define loss
                self.loss = tf.reduce_mean(tf.square(self.gt - self.est))/2

            with tf.name_scope("Opt"):
                #Define optimizer
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)
                self.optimizerFC = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss,
                        var_list=[self.W_conv6,
                            self.B_conv6,
                            self.W_fc1,
                            self.B_fc1,
                            self.W_fc2,
                            self.B_fc2,
                            self.W_fc3,
                            self.B_fc3,
                            self.W_fc4,
                            self.B_fc4]
                        )

        #Summaries
        tf.scalar_summary('l2 loss', self.loss)
        tf.histogram_summary('input', self.inputImage)
        tf.histogram_summary('gt', self.gt)
        tf.histogram_summary('conv1', self.h_pool1)
        tf.histogram_summary('conv2', self.h_pool2)
        tf.histogram_summary('conv3', self.h_conv3)
        tf.histogram_summary('conv4', self.h_conv4)
        tf.histogram_summary('conv5', self.h_pool5)
        tf.histogram_summary('conv6', self.h_pool6)
        tf.histogram_summary('fc1', self.h_fc1)
        tf.histogram_summary('fc2', self.h_fc2)
        tf.histogram_summary('fc3', self.h_fc3)
        tf.histogram_summary('est', self.est)
        tf.histogram_summary('w_conv1', self.W_conv1)
        tf.histogram_summary('b_conv1', self.B_conv1)
        tf.histogram_summary('w_conv2', self.W_conv2)
        tf.histogram_summary('b_conv2', self.B_conv2)
        tf.histogram_summary('w_conv3', self.W_conv3)
        tf.histogram_summary('b_conv3', self.B_conv3)
        tf.histogram_summary('w_conv4', self.W_conv4)
        tf.histogram_summary('b_conv4', self.B_conv4)
        tf.histogram_summary('w_conv5', self.W_conv5)
        tf.histogram_summary('b_conv5', self.B_conv5)
        tf.histogram_summary('w_conv6', self.W_conv6)
        tf.histogram_summary('b_conv6', self.B_conv6)
        tf.histogram_summary('w_fc1', self.W_fc1)
        tf.histogram_summary('b_fc1', self.B_fc1)
        tf.histogram_summary('w_fc2', self.W_fc2)
        tf.histogram_summary('b_fc2', self.B_fc2)
        tf.histogram_summary('w_fc3', self.W_fc3)
        tf.histogram_summary('b_fc3', self.B_fc3)
        tf.histogram_summary('w_fc4', self.W_fc4)
        tf.histogram_summary('b_fc4', self.B_fc4)

        #Define saver
        self.saver = tf.train.Saver()

    def initSess(self):
        self.sess.run(tf.initialize_all_variables())

    def writeSummary(self, summaryDir):
        self.mergedSummary = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(summaryDir + "/train", self.sess.graph)
        self.test_writer = tf.train.SummaryWriter(summaryDir + "/test")

    def closeSess(self):
        self.sess.close()

    def trainModel(self, numSteps, saveFile, pre=False, miniBatchSize = 32):
        #Define session
        for i in range(numSteps):
            #Get data from dataObj
            data = self.dataObj.getData(miniBatchSize)
            feedDict = {self.inputImage: data[0], self.gt: data[1], self.keep_prob: .5}
            #Run optimizer
            if(pre):
                self.sess.run(self.optimizerFC, feed_dict=feedDict)
            else:
                self.sess.run(self.optimizerAll, feed_dict=feedDict)
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.train_writer.add_summary(summary, self.timestep)
            self.timestep+=1
            if(i%self.progress == 0):
                print "Timestep ", self.timestep

        save_path = self.saver.save(self.sess, saveFile, global_step=self.timestep, write_meta_graph=False)
        print("Model saved in file: %s" % save_path)

    def evalModel(self, inData, inGt = None):
        (numData, ny, nx, nf) = inData.shape
        if(inGt != None):
            (numGt, drop) = inGt.shape
            assert(numData == numGt)
            feedDict = {self.inputImage: inData, self.gt: inGt, self.keep_prob: 1}
        else:
            feedDict = {self.inputImage: inData, self.keep_prob: 1}

        outVals = self.est.eval(feed_dict=feedDict, session=self.sess)
        if(inGt != None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)
        return outVals

    def evalModelBatch(self, miniBatchSize, inData, inGt=None):
        (numData, ny, nx, nf) = inData.shape
        if(inGt != None):
            (numGt, drop) = inGt.shape
            assert(numData == numGt)

        #Split up numData into miniBatchSize and evaluate est data
        tfInVals = np.zeros((miniBatchSize, ny, nx, nf))
        outData = np.zeros((numData, 1))

        #Ceil of numData/batchSize
        numIt = int(numData/miniBatchSize) + 1

        #Only write summary on first it

        startOffset = 0
        for it in range(numIt):
            #Calculate indices
            startDataIdx = startOffset
            endDataIdx = startOffset + miniBatchSize
            startTfValIdx = 0
            endTfValIdx = miniBatchSize

            #If out of bounds
            if(endDataIdx >= numData):
                #Calculate offset
                offset = endDataIdx - numData
                #Set endDataIdx to max value
                endDataIdx = numData
                #Set endTfValIdx to less than max value
                endTfValIdx -= offset

            tfInVals[startTfValIdx:endTfValIdx, :, :, :] = inData[startDataIdx:endDataIdx, :, :, :]
            feedDict = {self.inputImage: tfInVals, self.keep_prob: 1}
            tfOutVals = self.est.eval(feed_dict=feedDict, session=self.sess)
            outData[startDataIdx:endDataIdx, :] = tfOutVals[startTfValIdx:endTfValIdx, :]

            if(inGt != None and it == 0):
                tfInGt = inGt[startDataIdx:endDataIdx, :]
                summary = self.sess.run(self.mergedSummary, feed_dict={self.inputImage: tfInVals, self.gt: tfInGt, self.keep_prob: 1})
                self.test_writer.add_summary(summary, self.timestep)

            startOffset += miniBatchSize

        #Return output data
        return outData

    def loadModel(self, loadFile):
        self.saver.restore(self.sess, loadFile)
        print("Model %s loaded" % loadFile)

    #def saveModel(self, saveFile):
    #    #Save model


        #TO load:
        #saver.restore(sess, "/tmp/model.ckpt")
        #print("Model restored.")

