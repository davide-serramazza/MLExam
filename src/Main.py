#!/usr/bin/env python2

from Neural_network import *
import matplotlib.pyplot as plt


def main():
    dummy_test()

def dummy_test():
    arch = [2,2,2]
    neuronsType = [InputNeuron, SigmoidNeuron, SigmoidNeuron]
    network = Network(arch, neuronsType)
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
    network.dump_weights()

    network.layers[1].neurons[0].weights = [0.15, 0.2, 0.35]
    network.layers[1].neurons[1].weights = [0.25, 0.3, 0.35]
    network.layers[2].neurons[0].weights = [0.4, 0.45, 0.6]
    network.layers[2].neurons[1].weights = [0.5, 0.55, 0.6]
    print "new weights: "
    network.dump_weights()

    data = [[0.05, 0.1]]
    target = [[0.01, 0.99]]
    network.forward(pattern=data[0])
    delta_w, _,_ = network.back_propagation(target=target[0], eta=0.5)
    print "weights after a back propagation step:"
    network.update_weights(delta_w=delta_w)
    network.dump_weights()

    print "weights after one epoch of training:"
    network.layers[1].neurons[0].weights = [0.15, 0.2, 0.35]
    network.layers[1].neurons[1].weights = [0.25, 0.3, 0.35]
    network.layers[2].neurons[0].weights = [0.4, 0.45, 0.6]
    network.layers[2].neurons[1].weights = [0.5, 0.55, 0.6]
    network.train(data=data, targets=target, learning_rate=0.5, epochs=1, batch_size=len(data), momentum=0.0)
    network.dump_weights()

    # dump weights on file
    with open("weights.csv", "w") as file_weights:
        network.dump_weights(file_weights)
    with open("weights.csv", "r") as input_file:
        network.load_weights(input_file)
    network.dump_weights()

    arch = [2,2,1]
    neuronsType = [InputNeuron, SigmoidNeuron,SigmoidNeuron]
    network = Network(arch, neuronsType)
    datal = [[0,1],[0,0],[1,0],[1,1]]
    target = [0,1,0,1]
    losses = network.train(data=datal, targets=target, epochs=100, learning_rate=0.5,
                       batch_size=2,momentum=0.0)
    # 4. visualize how loss changes over time
    #    plots changes a lot for different runs
    plt.plot(range(len(losses)), losses)
    plt.xlabel("epochs")
    plt.ylabel("misClassification")
    plt.show()


    # last layer  0.3589 0.4086 -
    #             0.5113 0.5613 -
    # first layer 0.1497 0.1995 -
    #             0.2497 0.2995 -



if __name__ == '__main__':
    main()