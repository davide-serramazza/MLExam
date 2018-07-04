from Validation import *
import time
from utils import *
from grid_search import GridSearchSGDParams, grid_search_SGD
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

    # 4. define architecture and hyperparameters
    architecture = [[17, 20, 1]]
    neurons = [[InputNeuron, TanHNeuron, TanHNeuron]]
    momentum = [0.9]
    batch_size = [32, 64]
    learning_rate = [0.01]
    regularization = [0.0, 0.01]
    epoch = 300
    param = GridSearchSGDParams(learning_rate, momentum, batch_size,
                                  architecture, neurons, regularization, epoch)

    start_time = time.time()
    grid_search_SGD(param, lossObject, training_patterns, training_labels,
                validation_patterns, validation_labels, 5, "../grid_search_results/sgd/monk3/")
    elapsed_time = time.time() - start_time
    print "time in grid search:", elapsed_time


if __name__ == "__main__":
    main()
