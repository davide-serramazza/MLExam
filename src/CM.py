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


    ##' esempio
    arch = [17, 10, 1]
    neuronsType = [InputNeuron, TanHNeuron, TanHNeuron]
    network = Network(arch, neuronsType)

    print "\nBFGS\n"
    network.trainBFGS(training_patterns, training_labels, [],[],lossObject, 30)

    network = Network(arch, neuronsType)
    print "\nLBFGS\n"
    network.trainLBFGS(training_patterns, training_labels, [], [], lossObject, m=10, epochs=50)

### END EXAMPLE

if __name__ == '__main__':
    main()