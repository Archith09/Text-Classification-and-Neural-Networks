'''
    TEST SCRIPT FOR NEURAL NETWORKS
    AUTHOR Archith Shivanagere
'''

import numpy as np
from nn import NeuralNet

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    '''
        Main function to test Neural Networks
    '''
    filePathX = "data/digitsX.dat"
    file = open(filePathX,'r')
    allDataX = np.loadtxt(file, delimiter=',')

    X = allDataX

    filePathY = "data/digitsY.dat"
    file = open(filePathY,'r')
    allDataY = np.loadtxt(file)
    y = allDataY

    layers = np.array([25])
    
    modelNets = NeuralNet(layers, epsilon = 0.5, learningRate = 2.45, numEpochs = 1000)
    modelNets.fit(X, y)
    
    ypred_Nets = modelNets.predict(X)

    accuracyNets = accuracy_score(y, ypred_Nets)

    print "Neural Nets Accuracy = "+str(accuracyNets)

    filename = "Hidden_Layers.bmp"
    modelNets.visualizeHiddenNodes(filename)