'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from PIL import Image
from sklearn import preprocessing

class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate=2.45, numEpochs=100):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.values = None
        self.thetas = None
        self.lambdaC = 0.001
        self.classes_ = None
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        backwardPropogation = 1
        length = 0
        n, d = X.shape
        clf = preprocessing.LabelBinarizer()
        clf.fit(y)
        self.classes = clf.classes_
        ysize = clf.classes_.size
        ybinarized = clf.transform(y)
        self.thetas = dict()
        if(True):
            layersVector = np.concatenate((self.layers, [ysize]), axis = 0)
            layersVector = np.concatenate(([d], layersVector), axis = 0)
            range = layersVector.size - 1
            for flag in xrange(range):
                self.thetas[flag+1] = np.random.random_sample([layersVector[flag+1], layersVector[flag] + 1])*self.epsilon*2.0 - self.epsilon
        else:
            self.thetas[flag+1] = np.random.random_sample([d+1, ysize])*self.epsilon*2.0 - self.epsilon
        finalLayer = len(self.thetas)
        if length:
            print "Classes: ", ysize
            print "Instances: ", X.shape[0]
            print "Features: ", X.shape[1]
            print "Layers: ", layersVector
            print "Final Layer: ", finalLayer
            for index in xrange(finalLayer):
                print "theta",index+1," size: ", self.thetas[index+1].shape
            print "Now Starting Backward Propogation"
            
        if backwardPropogation:
            diffs = dir()
            gradientVar = dir()
            for index in xrange(self.numEpochs):
                self.forwardProp(X, self.thetas)
                diffs[finalLayer] = self.values[finalLayer][:,1:] - ybinarized
                for new in xrange(len(self.thetas)-1,-1,-1):
                    temp = new + 1
                    if(new<>0):
                        sgGrad = np.multiply(self.values[new][:,1:], (1.0 - self.values[new][:,1:]))
                        diffs[new] = np.multiply(np.dot(diffs[temp],self.thetas[temp][:,1:]),sgGrad)
                for new in xrange(len(self.thetas)):
                    temp = new + 1
                    gradientVar[temp] = np.dot(diffs[temp].T, self.values[new])
                    regularization = np.concatenate((np.zeros([self.thetas[temp].shape[0], 1]), self.thetas[temp][:,1:]), axis = 1)*self.lambdaC
                    a = X.shape[0]
                    gradientVar[temp] = (gradientVar[temp] / a + regularization)
                    self.thetas[temp] = self.thetas[temp] - self.learningRate * gradientVar[temp]
                    


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        self.forwardProp(X, self.thetas)
        final = len(self.thetas)
        return np.argmax(self.values[final][:,1:], axis = 1)
    
    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        hiddenLayers = dict()
        flag = 1
        #actual = self.thetas[flag][:,1:].shape
        shapes = self.thetas[flag][:,1:].shape[1]
        realShape = int(np.sqrt(shapes))
        range = self.thetas[flag].shape[0]
        for index in xrange(range):
            redo = self.thetas[flag][index,1:]
            redo = (redo - np.min(redo))
            redo = redo*255.0/np.max(redo)
            hiddenLayers[index] = redo.reshape((realShape, realShape))
        frames = self.thetas[flag].shape[0]
        frameNum = int(np.sqrt(frames))
        picture = np.zeros((frameNum*realShape, frameNum*realShape))
        tab = 0
        for index in xrange(frameNum):
            horizontal0 = index * realShape
            horizontal1 = (index+1) * realShape
            for new in xrange(frameNum):
                vertical0 = new * realShape
                vertical1 = (new+1) * realShape
                picture[horizontal0:horizontal1, vertical0:vertical1] = hiddenLayers[tab]
                tab = tab + 1
        display = Image.fromarray(picture.astype(np.uint8))
        display.show()
        display.save(filename)
        
    def sigmoid(self, z):
        '''
        Computers the sigmoid function 1/(1+exp(-z))
        '''
        
        return 1.0/(1.0 + np.exp(-z))
        
    def forwardProp(self, X, thetas):
        '''
        Implementing a private function as suggested in Homework file
        '''
        X = np.c_[np.ones(X.shape[0]), X]
        n, d = X.shape
        self.values = dict()
        self.values[0] = X
        self.values[1] = self.sigmoid(np.dot(X, thetas[1].T))
        a = self.values[1].shape[0]
        self.values[1] = np.c_[np.ones(a), self.values[1]]
        range = len(thetas) - 1
        for index in xrange(range):
            self.values[index + 2] = self.sigmoid(np.dot(self.values[index + 1], thetas[index + 2].T))
            self.values[index + 2] = np.c_[np.ones(self.values[index + 2].shape[0]), self.values[index + 2]]
