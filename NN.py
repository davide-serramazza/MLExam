#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:26:19 2017

@author: davide
"""
import numpy as np
import copy
import random
import abc
#superclass abstract Neuron
class Neuron(object):
    
    def __init__ (self):
        self.output = 0.0
    
    def activation_function(self,x):
        pass
    def activation_function_derivate(self):
        pass
    def getResult(self):
        return self.output
    
 #subclasses
class InputNeuron(Neuron):
    
    def activation_function(self,x):
        self.output = x
        return self.output
    def activation_function_derivate(self):
        return 1
    
    
class SigmoidNeuron (Neuron):
    
    def activation_function(self,x):
        self.output = 1/(1+np.exp(-x))
        return self.output
    def activation_function_derivate(self):
        return self.output*(1-self.output);
    
    
class BiasNeuron(Neuron):
    
    def __init__ (self):
        self.output = 1.0

    def activation_function(self,x):
        return self.output
    def activation_function_derivate(self):
        return 0.0
    
    
class OutputNeuron (Neuron):
    
    def activation_function(self,x):
        self.output = x
        return self.output

        
class Layer:
    
    def __init__(self,size,prevSize,neuron):
        # neuron's and weight's list
        self.neurons = []
        self.w = []
        #add hidden neurons
        for i in range (size):
            self.neurons.append( copy.deepcopy(neuron) )
        # add bias neuron
        self.neurons.append( BiasNeuron() )
        # add weights
        if prevSize != 0: #if is not an input layer
            for i in range (size):
                self.w.append([])
                for j in range (prevSize):
                    self.w[i].append( random.uniform(-1.0,1.0) )
        
        
class Network:
    
    def __init__(self,architecture):
        #input layer
        self.layers = [] 
        tmp = InputNeuron()
        l = Layer(architecture[0],0,tmp)
        self.layers.append(l)
        # hidden layers
        for i in range (1,len(architecture)-1):
            tmp = SigmoidNeuron()
            l = Layer (architecture[i], architecture[i-1],tmp)
            self.layers.append(l)
        # output layers
        tmp = OutputNeuron()
        size = len (architecture) -1 
        l = Layer(architecture[size],architecture[size-1],tmp)
        self.layers.append(l)
        
        
    
    
#test
arch = [3,5,2]
n = Network(arch)
print "layer's number" , len (n.layers)
for i in range (len (n.layers)):
    print "layer", i , "is size:" , len (n.layers[i].neurons)
print "activ foo for input neuron" , n.layers[0].neurons[0].activation_function(1.5)
print "activ foo for hidden neuron",  n.layers[1].neurons[0].activation_function(2.0)
print "activ foo's derivative for hidden neuron" , n.layers[1].neurons[0].activation_function_derivate()
print "activ foo for bias neuron" ,n.layers[1].neurons[5].activation_function(2.0)
print "activ foo's derivative for bias neuron", n.layers[1].neurons[5].activation_function_derivate()
print "activ foo for output neuron" , n.layers[2].neurons[0].activation_function(2.0)
print "wieght:"
for i in range (len (n.layers)):
    print n.layers[i].w
    print ""