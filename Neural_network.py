import numpy as np
from Layer import *


class Network:
    def __init__(self, architecture,i,t):
        # input layer
        self.layers = []
        self.input = i
        self.target = t
        self.output = []
        tmp = InputNeuron()
        l = Layer(architecture[0], 0, tmp)
        self.layers.append(l)
        # hidden layers
        for i in range(1, len(architecture) - 1):
            tmp = SigmoidNeuron()
            l = Layer(architecture[i], architecture[i - 1], tmp)
            self.layers.append(l)
        # output layers
        tmp = OutputNeuron()
        size = len(architecture) - 1
        l = Layer(architecture[size], architecture[size - 1], tmp)
        self.layers.append(l)
        
    def Forward(self):
        
        for i in range (len (self.layers[0].neurons)-1):
            # set first layer to input
            neuron = self.layers[0].neurons[i]
            neuron.activation_function( self.input[i] )
        
        #propagate result
        for i in range (1,len(self.layers)):
            for j in range (len (self.layers[i].neurons) -1 ): # -1 per escludere bias
                neuron = self.layers[i].neurons[j]
                netW = self.layers[i].w
                net = self.layers[i-1].getResults()
                arg = np.dot (netW[j] , net)
                self.layers[i].neurons[j].activation_function(arg)
        # setting output
        for i in range (len (self.layers[-1].neurons) -1 ):
            self.output.append (self.layers[-1].neurons[i].getResult())       
        
    
    def BackProp(self):
        #output layer
        tmp = len (self.layers) - 1 
        for i in range (len (self.layers[tmp].neurons)):
            # reference ti current analyzed neuron
            neuron = self.layers[tmp].neurons[i]
            oi = self.layers[tmp-1].nuerons
        return

    def train(self):
        # fit the data
        pass

    def predict(self):
        # predict target variable
        pass

    def serialize(self):
        # dump neural network weights to file
        # calls serialize on each layer
        pass
