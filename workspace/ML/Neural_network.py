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
        
        
    def BackProp(self,eta):
        #using as much as possible the nomenclature used in backPropaation lecture
        delta = []
        #output layer
        outputLayer = len (self.layers) - 1 
        deltaW = copy.deepcopy(self.layers[outputLayer].w) #vector in wich save what i'll add to last layer
        # reference to current analyzed neuron
        for i in range (len (self.layers[outputLayer].neurons) -1): #-1 due to exclude bias
            # current w
            for j in range (len (self.layers[outputLayer].w[i])):
                # oi = partial(currrent's neuron net)/partial current weight
                oi = self.layers[outputLayer-1].neurons[j].getResult()
               # print "oi" , oi
                # DF = partial(Error)/partial(input to neuron)
                Df = self.output[i] - self.target[i]
                #print "Df" , Df
                #Dneuron = partial (output's neuron)/partial (current neuron' snet)/ 
                Dneuron = self.layers[outputLayer].neurons[i].activation_function_derivate()
                if j == 0:
                    delta.append(Df*Dneuron)
                    tmp = Df*Dneuron
                else:
                    tmp = Df* Dneuron
                deltaW[i][j] = oi*tmp
        print "delta is:", delta
        print "last layer is:" , self.layers[2].w
        #hidden layers
        for i in range ( outputLayer -1 ,0,-1) :
            for j in range (len(self.layers[i].neurons) -1) : # .1 due to exclude bias
                for k in range (len(self.layers[i].w[j])):
                    sum = 0.0
                    for s in range ( len (self.layers[outputLayer].neurons ) - 1):
                        # tmp = parial(Error)/partial(neruon's net)
                        tmp = delta[s]
                        #Dnet = partial (net)/partial(o)
                        Dnet = self.layers[outputLayer].w[s][j]
                        sum += tmp*Dnet
                    #print "sum:" , sum
                    # Dout = partial(current neuron's out)/partial (current neuron's net)
                    Dout = self.layers[i].neurons[j].activation_function_derivate()
                    #print "Dout:", Dout
                    # Dnet = partial (currents neuron's net)/partial(current analyzerd weight's)
                    Dnet = self.layers[i-1].neurons[k].getResult()
                    #print "Dnet:" , Dnet
                    self.layers[i].w[j][k] -= eta*sum*Dout*Dnet
        for i in range (len(deltaW)):
            for j in range(len(deltaW[i])):
                self.layers[outputLayer].w[i][j] -=eta*deltaW[i][j]

                
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
