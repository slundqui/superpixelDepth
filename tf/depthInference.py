import pdb
import numpy as np
import tensorflow as tf

weightInitStd = .001
biasInitConst = .001

#For initializing weights and biases
def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=weightInitStd)
   return tf.Variable(initial)

def bias_variable(shape):
   initial = tf.constant(.1, shape=shape)
   return tf.Variable(initial)

def node_variable(shape):
   return tf.placeholder("float", shape=shape)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')

class unaryDepthInference:
    #Initialize tf parameters here
    miniBatchSize = 128
    progress = 100
    learningRate = 1e-5
    lossVals = []

    def __init__(self, dataObj):
        self.sess = tf.InteractiveSession()
        self.dataObj = dataObj
        self.buildModel()

    def buildModel(self):
        inputShape = self.dataObj.inputShape
        #Get convolution variables as placeholders
        self.inputImage = node_variable([self.miniBatchSize, inputShape[0], inputShape[1], inputShape[2]])
        self.gt = node_variable([self.miniBatchSize, 1])
        #Model variables for convolutions

        #First conv layer is 11x11, 3 input channels into 64 output channels
        self.W_conv1 = weight_variable([11, 11, 3, 64])
        self.B_conv1 = bias_variable([64])
        self.h_conv1 = conv2d(self.inputImage, self.W_conv1) + self.B_conv1
        #relu is communative op, so do relu after pool for efficiency
        self.h_pool1 = tf.nn.relu(maxpool_2x2(self.h_conv1))

        #Second conv layer is 5x5 conv, into 256 output channels
        self.W_conv2 = weight_variable([5, 5, 64, 256])
        self.B_conv2 = bias_variable([256])
        self.h_conv2 = conv2d(self.h_pool1, self.W_conv2) + self.B_conv2
        self.h_pool2 = tf.nn.relu(maxpool_2x2(self.h_conv2))

        #Third layer is 3x3 conv into 256 output channels
        #No pooling
        self.W_conv3 = weight_variable([3, 3, 256, 256])
        self.B_conv3 = bias_variable([256])
        self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3) + self.B_conv3)

        #Fourth layer is 3x3 conv into 256 output channels
        #No pooling
        self.W_conv4 = weight_variable([3, 3, 256, 256])
        self.B_conv4 = bias_variable([256])
        self.h_conv4 = tf.nn.relu(conv2d(self.h_conv3, self.W_conv4) + self.B_conv4)

        #Fifth layer is 3x3 conv into 256 output channels
        #with pooling
        self.W_conv5 = weight_variable([3, 3, 256, 256])
        self.B_conv5 = bias_variable([256])
        self.h_conv5 = conv2d(self.h_conv4, self.W_conv5) + self.B_conv5
        self.h_pool5 = tf.nn.relu(maxpool_2x2(self.h_conv5))

        #Next is 3 fully connected layers
        #We should have downsampled by 8 at this point
        #fc1 should have 4096 channels
        numInputs = (inputShape[0]/8) * (inputShape[1]/8) * 256
        self.W_fc1 = weight_variable([numInputs, 4096])
        self.B_fc1 = bias_variable([4096])
        h_pool5_flat = tf.reshape(self.h_pool5, [-1, numInputs])
        self.h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, self.W_fc1) + self.B_fc1)

        #fc2 should have 128 channels
        self.W_fc2 = weight_variable([4096, 128])
        self.B_fc2 = bias_variable([128])
        self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2) + self.B_fc2)

        #fc3 should have 16 channels
        #fc3 also uses a sigmoid function
        self.W_fc3 = weight_variable([128, 16])
        self.B_fc3 = bias_variable([16])
        self.h_fc3 = tf.sigmoid(tf.matmul(self.h_fc2, self.W_fc3) + self.B_fc3)

        #Finally, fc4 condenses into 1 output value
        self.W_fc4 = weight_variable([16, 1])
        self.B_fc4 = bias_variable([1])
        self.est = tf.matmul(self.h_fc3, self.W_fc4) + self.B_fc4

        #Define loss
        self.loss = tf.nn.l2_loss(self.gt - self.est)

        #Define optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)

        #Define saver
        self.saver = tf.train.Saver()

        self.sess.run(tf.initialize_all_variables())

    def trainModel(self, numSteps, saveFile):
        data = self.dataObj.getData(self.miniBatchSize)

        for i in range(numSteps):
            feedDict = {self.inputImage: data[0], self.gt: data[1]}
            if i%progress:
                self.lossVals.append(self.loss.eval(feed_dict=feedDict))
                print("step %d, training loss %g"%(i, self.lossVals[-1]))

    def clearLoss(self):
        self.lossVals = []

    def plotLoss(self):
        plt.plot(lossVals)
        plt.show()

    def saveModel(self, saveFile):
        #Save model
        save_path = self.saver.save(self.sess, saveFile)
        print("Model saved in file: %s" % save_path)


        #TO load:
        #saver.restore(sess, "/tmp/model.ckpt")
        #print("Model restored.")

