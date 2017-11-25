import numpy as np
from Layer import *


class Network:
    def __init__(self, architecture, i, t):
        # input layer
        self.layers = []
        self.input = i
        self.target = t
        self.output = []
        inputNeuron = InputNeuron()
        l = Layer(architecture[0], 0, inputNeuron)
        self.layers.append(l)
        # hidden layers
        for i in range(1, len(architecture) - 1):
            neuron = SigmoidNeuron(len_weights=len(self.layers[i-1].neurons))
            layer = Layer(architecture[i], architecture[i - 1], neuron)
            self.layers.append(layer)
        # output layers
        outputNeuron = OutputNeuron(len_weights=len(self.layers[-1].neurons))
        size = len(architecture) - 1
        layer = Layer(architecture[size], architecture[size - 1], outputNeuron)
        self.layers.append(layer)
        
    def Forward(self):
        
        for i in range (len (self.layers[0].neurons)-1):
            # set first layer to input
            neuron = self.layers[0].neurons[i]
            neuron.activation_function( self.input[i] )
        
        #propagate result
        for i in range (1,len(self.layers)):
            for j in range (len (self.layers[i].neurons) -1 ):  # exclude bias neuron
                neuron = self.layers[i].neurons[j]
                weights = neuron.weights
                input_x = self.layers[i-1].getOutput()
                scores = np.dot (weights , input_x)
                neuron.activation_function(scores)
        # setting output
        last_layer = self.layers[-1]
        for i in range (len (last_layer.neurons) -1 ):
            self.output.append (last_layer.neurons[i].getOutput())
        
    
    def BackProp(self):
        #output layer
        tmp = len (self.layers) - 1 
        for i in range (len (self.layers[tmp].neurons)):
            # reference to current analyzed neuron
            neuron = self.layers[tmp].neurons[i]
            oi = self.layers[tmp-1].neurons
        return

    def train(self):
        # fit the data
        # for each iteration/epoch
        #   self.forward()
        #   self.backprop()
        pass

    def predict(self, data):
        # predict target variable
        # scores = forward(data)
        pass

    def serialize(self):
        # dump neural network weights to file
        # calls serialize on each layer
        pass
