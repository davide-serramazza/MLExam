import pandas as pd
from Validation import  *
from monk_benchmark import *
from Neural_network import *

def main():
    train_file = "../monk_datasets/monks-1.train"

        # 1. load dataset
    columns = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'id']
    train_data = pd.read_csv(train_file, delim_whitespace=True, header=None)
    train_data.columns = columns

    # 2. hold out
    frac = 0.7
    training_set, validation_set = holdout(frac, train_data)

    # 3. decode patterns and transform targets
    encoding = [3, 3, 2, 3, 4, 2]
    features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
    training_patterns, validation_patterns = decode_patterns(encoding, features, training_set, validation_set)
    training_labels, validation_labels = transform_labels(training_set, validation_set)

    lossObject = SquaredError("tangentH")

    arch = [17, 10, 1]
    neuronsType = [InputNeuron, TanHNeuron, TanHNeuron]
    network = Network(arch, neuronsType)
    #network.layers[1].neurons[0].weights = np.asarray([0.15, 0.2, 0.35])
    #network.layers[1].neurons[1].weights = np.asarray([0.25, 0.3, 0.35])
    #network.layers[2].neurons[0].weights = np.asarray([0.4, 0.45, 0.6])
    #network.layers[2].neurons[1].weights = np.asarray([0.5, 0.55, 0.6])

    #data = [[0.05, 0.1]]
    #target = [[0.01, 0.99]]
    #network.forward(pattern=data[0])
    #print network.predict(data)

    network.trainBFGS(training_patterns, training_labels, training_patterns, training_labels,lossObject, 400)
    scores = network.predict(training_patterns)
    print "error:", np.sum(np.square(np.array(scores) - np.array(training_labels))) / len(training_labels)
    """
    net = Network([17,10,1],[InputNeuron,TanHNeuron,TanHNeuron])

    net.trainBFGS(training_patterns,training_labels,validation_patterns,validation_labels,lossObject,5)

    tmp = net.predict(training_patterns)
    print tmp
"""

if __name__ == '__main__':
    main()