import pandas as pd
from Neural_network import *
import matplotlib.pyplot as plt
from Validation import *

def decode(data,encoding):
    """
    Decode examples encoded with 1-of-k

    :param data: vector of examples to decode
    :param encoding: 1-of-k encoding used
    :return: decoded data
    """
    ris = []
    for i in range (len(data)):
        for j in range(1,encoding[i]+1):
            if j==data[i]:
                ris.append(1)
            else:
                ris.append(0)
    return ris

def transform_target(l):
    """
    transform specific negative example's target from 0 to -1
    (needed if using tanH as output)

    :param l: vector containing targets
    :return: transformed vector targets
    """
    res = []
    for i in l:
        if i==0:
            res.append(np.array([-1]))
        else:
            res.append(np.array([1]))
    return res

def transform_labels(training_set, validation_set):
    training_labels = transform_target(training_set["label"].values)
    validation_labels = transform_target(validation_set["label"].values)
    return training_labels, validation_labels


def decode_patterns(encoding, features, training_set, validation_set):
    training_patterns = [decode(pattern, encoding) for pattern in training_set[features].values]
    validation_patterns = [decode(pattern, encoding) for pattern in validation_set[features].values]
    return training_patterns, validation_patterns

def main():
    train_file = "../monk_datasets/monks-3.train"

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

    # validation: define hyper-parameters to test
    architecture = [[17, 10, 1]]
    neurons = [[InputNeuron, TanHNeuron, TanHNeuron]] #[[InputNeuron, TanHNeuron, TanHNeuron], [InputNeuron, TanHNeuron, TanHNeuron,TanHNeuron] ]
    momentum = [0.4]  # [0.4, 0.5, 0.6]
    batch_size = [10]
    learning_rate = [0.15]  #[0.15, 0.2, 0.25]
    regularization = [0.05]  # [0.0025, 0.005, 0.001]
    epoch = 50
    param = grid_search_parameter(learning_rate, momentum, batch_size,
                                  architecture, neurons, regularization, epoch)

    grid_search(param, lossObject, training_patterns, training_labels,
                validation_patterns, validation_labels, 5, "../image/aa-reg")


if __name__ == "__main__":
    main()
