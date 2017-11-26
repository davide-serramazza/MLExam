#!/usr/bin/env python2

from Neural_network import *
    
    
def main():
    dummy_test()

def dummy_test():
    arch = [2,2,2]
    neuronsType = [InputNeuron, SigmoidNeuron, OutputNeuron]
    # actually only one example
    traningSet = []
    traningSet.append ([0.05,0.1])
    target = []
    target.append([0.01,0.99])
    network = Network(arch, neuronsType,traningSet,target)
    print "layer's number", len(network.layers)
    for i in range(len(network.layers)):
        print "layer", i , "is size:", len (network.layers[i].neurons)
        
    print "activ foo for input neuron:" , network.layers[0].neurons[0].activation_function(1.5) , "with argument", 1.5
    print "activ foo for hidden neuron:",  network.layers[1].neurons[0].activation_function(2.0) , "with argument", 2.0
    print "activ foo's derivative for hidden neuron:" , network.layers[1].neurons[0].activation_function_derivative()
    print "activ foo for bias neuron:" ,network.layers[1].neurons[2].activation_function(2.0), "with argument", 2.0
    print "activ foo's derivative for bias neuron:", network.layers[1].neurons[2].activation_function_derivative()
    print "activ foo for output neuron:" , network.layers[2].neurons[0].activation_function(2.0), "with argument", 2.0
    print "weights:"
    for layer in network.layers:
        print [neuron.weights for neuron in layer.neurons if
               not (isinstance(neuron, InputNeuron) or isinstance(neuron, BiasNeuron))]
        print ""

    print "new weights:"
    for layer in network.layers:
        print [neuron.weights for neuron in layer.neurons if not ( isinstance(neuron, InputNeuron) or isinstance(neuron, BiasNeuron))]
        print ""        
    network.forward()
    print "output's forward", network.output
    network.BackProp(0.5,0)
    print "weights after a backProp step:"
    for l in network.layers:
        for n in l.neurons:
            if isinstance(n, SigmoidNeuron):
                print n.weights
                
    
    
if __name__ == '__main__':
    main()