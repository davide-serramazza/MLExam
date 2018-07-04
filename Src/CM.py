from monk_benchmark import *
from Validation import *
from Neural_network import *
from grid_search import GridSearchLBFGSParams, grid_search_LBFGS
import time

def main():
    train_file = "../Data/monk_datasets/monks-3.train"

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

    # 4. define architecture and hyper parameter
    arch = [[17,10,1]]
    neuronsType = [[InputNeuron, TanHNeuron, TanHNeuron]]

    #network = Network(arch, neuronsType)
    c_1 = [0.0001]
    c_2 = [0.9]
    theta = [0.9]
    regularization = [0.0]
    m = [50]
    epochs = 20
    lossObject = SquaredError("tangentH")
    parameter = GridSearchLBFGSParams(c_1,c_2,theta,regularization,m,epochs,arch,neuronsType)
    # perform grid search

    tic = time.time()
    grid_search_LBFGS(parameter,lossObject,training_patterns,training_labels,validation_patterns,validation_labels,
                   n_trials=3, save_in_dir="../temp/monk_1-")

    toc = time.time()
    print "time in grid search: %.2f seconds" % (toc-tic)


if __name__ == '__main__':
    main()