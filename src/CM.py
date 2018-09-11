from monk_benchmark import *
from Validation import *
from Neural_network import *
from grid_search import *
import time

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

    # 4. define architecture and hyper parameter
    arch = [[17,20,1] , [17,15,1]]
    neuronsType = [[InputNeuron, TanHNeuron, TanHNeuron], [InputNeuron, TanHNeuron, TanHNeuron]]

    c_1 = [0.0001]
    c_2 = [0.9]
    theta = [0.9]
    reguralization = [0.00001, 0.0001, 0.001, 0.01]
    m = [40]
    epochs = 20
    epsilon = [0.001]
    lossObject = SquaredError("tangentH")
    parameter = GridSearchLBFGSParams(c_1,c_2,theta,reguralization, epsilon, m,epochs,arch,neuronsType)
    # perform grid search
    #grid_search_LBFGS(parameter,lossObject,training_patterns,training_labels,validation_patterns,validation_labels,
    #               n_trials=3,save_in_dir="../temp/new-",)

    # try BFGS
    network = Network([17, 20, 10, 1], [InputNeuron, TanHNeuron, TanHNeuron, TanHNeuron])
    loss_tr, miss_tr, loss_vl, miss_vl = network.train_BFGS(training_patterns, training_labels,
                                                 validation_patterns, validation_labels,
                                                 theta=0.5, c_1=0.0001, c_2=0.9, lossObject=lossObject,
                                                 regularization=0.0, epochs=50, epsilon=0.001)

    #loss_tr, miss_tr, loss_vl, miss_vl = network.trainLBFGS(training_patterns, training_labels,
    #                                             validation_patterns, validation_labels,
    #                                             theta=0.9, c_1=0.0001, c_2=0.9, lossObject=lossObject,
    #                                             regularization=0.01, epochs=10, epsilon=0.001, m=10)

    plot_train_test_learning_curve(loss_vl, loss_tr, miss_vl, miss_tr)

if __name__ == '__main__':
    main()