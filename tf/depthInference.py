import pdb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

weightInitStd = .001
biasInitConst = .001

#For initializing weights and biases
def weight_variable(shape, inName):
   initial = tf.truncated_normal(shape, stddev=weightInitStd, name=inName)
   return tf.Variable(initial)

def bias_variable(shape, inName):
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
    miniBatchSize = 32
    progress = 10
    learningRate = 1e-5
    lossVals = []

    def __init__(self, dataObj):
        self.dataObj = dataObj
        self.sess = tf.Session()
        self.buildModel()

    def buildModel(self):
        with tf.name_scope("inputOps"):
            inputShape = self.dataObj.inputShape
            #Get convolution variables as placeholders
            self.inputImage = node_variable([self.miniBatchSize, inputShape[0], inputShape[1], inputShape[2]], "inputImage")
            self.gt = node_variable([self.miniBatchSize, 1], "gt")
            #Model variables for convolutions

        with tf.name_scope("Conv1Ops"):
            #First conv layer is 11x11, 3 input channels into 64 output channels
            self.W_conv1 = weight_variable([11, 11, 3, 64], "w_conv1")
            self.B_conv1 = bias_variable([64], "b_conv1")
            self.h_conv1 = conv2d(self.inputImage, self.W_conv1, "conv1") + self.B_conv1
            #relu is communative op, so do relu after pool for efficiency
            self.h_pool1 = tf.nn.relu(maxpool_2x2(self.h_conv1, "pool1"), name="relu1")

        with tf.name_scope("Conv2Ops"):
            #Second conv layer is 5x5 conv, into 256 output channels
            self.W_conv2 = weight_variable([5, 5, 64, 256], "w_conv2")
            self.B_conv2 = bias_variable([256], "b_conv2")
            self.h_conv2 = conv2d(self.h_pool1, self.W_conv2, "conv2") + self.B_conv2
            self.h_pool2 = tf.nn.relu(maxpool_2x2(self.h_conv2, "pool2"), name="relu2")

        #Third layer is 3x3 conv into 256 output channels
        #No pooling
        with tf.name_scope("Conv3Ops"):
            self.W_conv3 = weight_variable([3, 3, 256, 256], "w_conv3")
            self.B_conv3 = bias_variable([256], "b_conv3")
            self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3, "conv3") + self.B_conv3, name="relu3")

        #Fourth layer is 3x3 conv into 256 output channels
        #No pooling
        with tf.name_scope("Conv4Ops"):
            self.W_conv4 = weight_variable([3, 3, 256, 256], "w_conv4")
            self.B_conv4 = bias_variable([256], "b_conv4")
            self.h_conv4 = tf.nn.relu(conv2d(self.h_conv3, self.W_conv4, "conv4") + self.B_conv4, name="relu4")

        #Fifth layer is 3x3 conv into 256 output channels
        #with pooling
        with tf.name_scope("Conv5Ops"):
            self.W_conv5 = weight_variable([3, 3, 256, 256], "w_conv5")
            self.B_conv5 = bias_variable([256], "b_conv5")
            self.h_conv5 = conv2d(self.h_conv4, self.W_conv5, "conv5") + self.B_conv5
            self.h_pool5 = tf.nn.relu(maxpool_2x2(self.h_conv5, "pool5"), name="relu5")

        #6th layer (not in paper) is 3x3 conv into 256 output channels
        #with pooling
        with tf.name_scope("Conv6Ops"):
            self.W_conv6 = weight_variable([3, 3, 256, 256], "w_conv6")
            self.B_conv6 = bias_variable([256], "b_conv6")
            self.h_conv6 = conv2d(self.h_pool5, self.W_conv6, "conv6") + self.B_conv6
            self.h_pool6 = tf.nn.relu(maxpool_2x2(self.h_conv6, "pool6"), name="relu6")

        #Next is 3 fully connected layers
        #We should have downsampled by 8 at this point
        #fc1 should have 4096 channels
        numInputs = (inputShape[0]/16) * (inputShape[1]/16) * 256
        with tf.name_scope("FC1"):
            self.W_fc1 = weight_variable([numInputs, 4096], "w_fc1")
            self.B_fc1 = bias_variable([4096], "b_fc1")
            h_pool6_flat = tf.reshape(self.h_pool6, [-1, numInputs], name="pool6_flat")
            self.h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat, self.W_fc1, name="fc1") + self.B_fc1, "fc1_relu")

        #fc2 should have 128 channels
        with tf.name_scope("FC2"):
            self.W_fc2 = weight_variable([4096, 128], "w_fc2")
            self.B_fc2 = bias_variable([128], "b_fc2")
            self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2, name="fc2") + self.B_fc2, "fc2_relu")

        #fc3 should have 16 channels
        #fc3 also uses a sigmoid function
        with tf.name_scope("FC3"):
            self.W_fc3 = weight_variable([128, 16], "w_fc3")
            self.B_fc3 = bias_variable([16], "b_fc3")
            self.h_fc3 = tf.sigmoid(tf.matmul(self.h_fc2, self.W_fc3, name="fc3") + self.B_fc3, "fc3_relu")

        #Finally, fc4 condenses into 1 output value
        with tf.name_scope("FC4"):
            self.W_fc4 = weight_variable([16, 1], "w_fc4")
            self.B_fc4 = bias_variable([1], "b_fc4")
            self.est = tf.matmul(self.h_fc3, self.W_fc4, name="est") + self.B_fc4

        with tf.name_scope("Loss"):
            #Define loss
            self.loss = tf.reduce_mean(tf.square(self.gt - self.est))/2

        with tf.name_scope("Train"):
            #Define optimizer
            self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)

        with tf.name_scope("Saver"):
            #Define saver
            self.saver = tf.train.Saver()


    def initSess(self):
        self.sess.run(tf.initialize_all_variables())

    def writeSummary(self, summaryFilename):
        tf.train.SummaryWriter(summaryFilename, self.sess.graph)
        tf.train.write_graph(sess.graph_def, summaryFilename, "unaryDepthInference.pb", False)

    def closeSess(Self):
        self.sess.close()

    def trainModel(self, numSteps, saveFile):
        #Define session
        for i in range(numSteps):
            #Get data from dataObj
            data = self.dataObj.getData(self.miniBatchSize)
            feedDict = {self.inputImage: data[0], self.gt: data[1]}
            #Run optimizer
            self.sess.run(self.optimizer, feed_dict=feedDict)
            #Progress step printout
            if i%self.progress == 0:
                self.lossVals.append(self.loss.eval(feed_dict=feedDict, session=self.sess))
                print("step %d, training loss %g"%(i, self.lossVals[-1]))
        save_path = self.saver.save(self.sess, saveFile)
        print("Model saved in file: %s" % save_path)

    def evalModel(self, inData):
        (numData, ny, nx, nf) = inData.shape

        #Split up numData into miniBatchSize and evaluate est data
        tfInVals = np.zeros((self.miniBatchSize, ny, nx, nf))
        outData = np.zeros((numData, 1))

        #Ceil of numData/batchSize
        numIt = int(numData/self.miniBatchSize) + 1

        startOffset = 0
        for it in range(numIt):
            #Calculate indices
            startDataIdx = startOffset
            endDataIdx = startOffset + self.miniBatchSize
            startTfValIdx = 0
            endTfValIdx = self.miniBatchSize

            #If out of bounds
            if(endDataIdx >= numData):
                #Calculate offset
                offset = endDataIdx - numData
                #Set endDataIdx to max value
                endDataIdx = numData
                #Set endTfValIdx to less than max value
                endTfValIdx -= offset

            tfInVals[startTfValIdx:endTfValIdx, :, :, :] = inData[startDataIdx:endDataIdx, :, :, :]
            feedDict = {self.inputImage: tfInVals}
            tfOutVals = self.est.eval(feed_dict=feedDict, session=self.sess)
            outData[startDataIdx:endDataIdx, :] = tfOutVals[startTfValIdx:endTfValIdx, :]

            startOffset += self.miniBatchSize

        #Return output data
        return outData

    def loadModel(self, loadFile):
        self.saver.restore(self.sess, loadFile)
        print("Model %s loaded" % loadFile)

    def clearLoss(self):
        self.lossVals = []


    #def saveModel(self, saveFile):
    #    #Save model


        #TO load:
        #saver.restore(sess, "/tmp/model.ckpt")
        #print("Model restored.")

