#!/usr/bin/env python2

from Neural_network import *
    
    
def main():
    dummy_test()

def dummy_test():
    arch = [3,5,2]
    network = Network(arch)
    print "layer's number", len(network.layers)
    for i in range(len(network.layers)):
        print "layer", i , "is size:", len (network.layers[i].neurons)
        print "activ foo for input neuron" , network.layers[0].neurons[0].activation_function(1.5)
        print "activ foo for hidden neuron",  network.layers[1].neurons[0].activation_function(2.0)
        print "activ foo's derivative for hidden neuron" , network.layers[1].neurons[0].activation_function_derivate()
        print "activ foo for bias neuron" ,network.layers[1].neurons[5].activation_function(2.0)
        print "activ foo's derivative for bias neuron", network.layers[1].neurons[5].activation_function_derivate()
        print "activ foo for output neuron" , network.layers[2].neurons[0].activation_function(2.0)
    print "weights:"
    for i in range(len (network.layers)):
        print network.layers[i].w
        print ""


if __name__ == '__main__':
    main()