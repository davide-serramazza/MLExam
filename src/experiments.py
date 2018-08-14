from Validation import *
from grid_search import *
from Neural_network import *
import sys

# files training set
monk1_train_file = "../Data/monk_datasets/monks-1.train"
monk2_train_file = "../Data/monk_datasets/monks-2.train"
monk3_train_file = "../Data/monk_datasets/monks-3.train"
cup_train_file = "../Data/MLCup/ML-CUP17-TR_shuffled.csv"

# save experiments in
sgd_dir = "../grid_search_results/sgd/"
bfgs_dir = "../grid_search_results/bfgs/"
lbfgs_dir = "../grid_search_results/lbfgs/"

# subdirectories to save experiements
monk1 = "monk1/reg_"
monk2 = "monk2/reg_"
monk3 = "monk3/reg_"
cup = "cup/reg_"

# monk dataset stuff
encoding = [3, 3, 2, 3, 4, 2]
features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
columns = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'id']
frac = 0.7

# cup dataset stuff
features_col = ["input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9", "input10"]
targets_col = ["target_x", "target_y"]
frac_cup = 0.8


def sgd_training_monk(train_file, grid_search_param, save_dir):
    # 1. load dataset
    train_data = pd.read_csv(train_file, delim_whitespace=True, header=None)
    train_data.columns = columns
    # 2. hold out
    training_set, validation_set = holdout(frac, train_data)
    # 3. decode patterns and transform targets
    training_patterns, validation_patterns = decode_patterns(encoding, features, training_set, validation_set)
    training_labels, validation_labels = transform_labels(training_set, validation_set)

    grid_search_SGD(grid_search_param, SquaredError("tangentH"), training_patterns, training_labels,
                    validation_patterns, validation_labels, n_trials=5, save_in_dir=save_dir)


def lbfgs_training_monk(train_file, grid_search_param, save_dir):
    # 1. load dataset
    train_data = pd.read_csv(train_file, delim_whitespace=True, header=None)
    train_data.columns = columns
    # 2. hold out
    training_set, validation_set = holdout(frac, train_data)
    # 3. decode patterns and transform targets
    training_patterns, validation_patterns = decode_patterns(encoding, features, training_set, validation_set)
    training_labels, validation_labels = transform_labels(training_set, validation_set)

    grid_search_LBFGS(grid_search_param, SquaredError("tangentH"), training_patterns, training_labels,
                      validation_patterns, validation_labels,
                      n_trials=5, save_in_dir=save_dir)


def bfgs_training_monk(train_file, grid_search_param, save_dir):
    # 1. load dataset
    train_data = pd.read_csv(train_file, delim_whitespace=True, header=None)
    train_data.columns = columns
    # 2. hold out
    training_set, validation_set = holdout(frac, train_data)
    # 3. decode patterns and transform targets
    training_patterns, validation_patterns = decode_patterns(encoding, features, training_set, validation_set)
    training_labels, validation_labels = transform_labels(training_set, validation_set)

    grid_search_BFGS(grid_search_param, SquaredError("tangentH"), training_patterns, training_labels,
                      validation_patterns, validation_labels,
                      n_trials=5, save_in_dir=save_dir)


def read_cup_data():
    df = pd.read_csv(cup_train_file, comment='#')
    # 2. divide pattern and targets
    patterns, labels = divide_patterns_labels(df, features_col, targets_col)
    # 4. divide in tr, vl and ts set
    first_partition_patterns, first_partition_labels, test_patterns, test_targets = holdout_cup(patterns,
                                                                                                labels, frac_cup)
    tr_patterns, tr_targets, vl_patterns, vl_targets = holdout_cup(first_partition_patterns
                                                                   , first_partition_labels, frac_cup)
    return tr_patterns, tr_targets, vl_patterns, vl_targets


def sgd_all_grid_search():
    global tr_patterns, tr_targets, vl_patterns, vl_targets
    param_m1 = GridSearchSGDParams(learning_rate=learning_rate,
                                momentum=momentum, batch_size=batch_size, architecture=architecture_monk1,
                                neurons=neurons_monk1, regularization=regularization, epoch=epochs_sgd)

    param_m2 = GridSearchSGDParams(learning_rate, momentum, batch_size, architecture_monk2, neurons_monk2,
                                   regularization=regularization, epoch=epochs_sgd)

    param_m3 = GridSearchSGDParams(learning_rate, momentum, batch_size, architecture_monk3, neurons_monk3,
                                   regularization=regularization, epoch=epochs_sgd)

    param_cup = GridSearchSGDParams(learning_rate, momentum, batch_size,
                                    architecture=architecture_cup,
                                    neurons=neurons_cup,
                                    regularization=regularization, epoch=epochs_sgd)
    #### SGD - MONK 1 ###
    print "\nSGD-MONK1\n"
    sgd_training_monk(train_file=monk1_train_file, grid_search_param=param_m1, save_dir=sgd_dir + monk1)
    #### SGD - MONK 2 ###
    print "\nSGD-MONK2\n"
    sgd_training_monk(train_file=monk2_train_file, grid_search_param=param_m2, save_dir=sgd_dir + monk2)
    #### SGD - MONK 3 ###
    print "\nSGD-MONK3\n"
    sgd_training_monk(train_file=monk3_train_file, grid_search_param=param_m3, save_dir=sgd_dir + monk3)
    #### SGD - CUP ###
    print "\nSGD-CUP\n"
    # 1. read dataset
    tr_patterns, tr_targets, vl_patterns, vl_targets = read_cup_data()
    grid_search_SGD(param_cup, EuclideanError(), tr_patterns, tr_targets,
                    vl_patterns, vl_targets, n_trials=5, save_in_dir=sgd_dir + cup)


def lbfgs_all_grid_search():
    global tr_patterns, tr_targets, vl_patterns, vl_targets

    param_lbfgs_m1 = GridSearchLBFGSParams(c_1=c_1, c_2=c_2, theta=theta, regularization=regularization, m=m_monk,
                                           epsilon=epsilon, epoch=epochs_lbfgs, architecture=architecture_monk1,
                                           neurons=neurons_monk1)

    param_lbfgs_m2 = GridSearchLBFGSParams(c_1=c_1, c_2=c_2, theta=theta, regularization=regularization, m=m_monk,
                                           epsilon=epsilon, epoch=epochs_lbfgs, architecture=architecture_monk2,
                                           neurons=neurons_monk2)

    param_lbfgs_m3 = GridSearchLBFGSParams(c_1=c_1, c_2=c_2, theta=theta, regularization=regularization, m=m_monk,
                                           epsilon=epsilon, epoch=epochs_lbfgs, architecture=architecture_monk3,
                                           neurons=neurons_monk3)

    param_lbfgs_cup = GridSearchLBFGSParams(c_1=c_1, c_2=c_2, theta=theta, regularization=regularization, m=m_cup,
                                            epsilon=epsilon, epoch=epochs_lbfgs, architecture=architecture_cup,
                                            neurons=neurons_cup)
    print "\nLBFGS-MONK1\n"
    lbfgs_training_monk(train_file=monk1_train_file, grid_search_param=param_lbfgs_m1, save_dir=lbfgs_dir + monk1)
    print "\nLBFGS-MONK2\n"
    lbfgs_training_monk(train_file=monk2_train_file, grid_search_param=param_lbfgs_m2, save_dir=lbfgs_dir + monk2)
    print "\nLBFGS-MONK3\n"
    lbfgs_training_monk(train_file=monk3_train_file, grid_search_param=param_lbfgs_m3, save_dir=lbfgs_dir + monk3)
    print "\nLBFGS-CUP\n"
    tr_patterns, tr_targets, vl_patterns, vl_targets = read_cup_data()
    grid_search_LBFGS(param_lbfgs_cup, EuclideanError(), tr_patterns, tr_targets,
                      vl_patterns, vl_targets,
                      n_trials=5, save_in_dir=lbfgs_dir + cup)


def bfgs_all_grid_search():
    global tr_patterns, tr_targets, vl_patterns, vl_targets

    param_bfgs_m1 = GridSearchBFGSParams(c_1=c_1, c_2=c_2, theta=theta, regularization=regularization, epsilon=epsilon,
                                         epoch=epochs_bfgs, architecture=architecture_monk1, neurons=neurons_monk1)

    param_bfgs_m2 = GridSearchBFGSParams(c_1=c_1, c_2=c_2, theta=theta, regularization=regularization, epsilon=epsilon,
                                      epoch=epochs_bfgs, architecture=architecture_monk2, neurons=neurons_monk2)

    param_bfgs_m3 = GridSearchBFGSParams(c_1=c_1, c_2=c_2, theta=theta, regularization=regularization, epsilon=epsilon,
                                          epoch=epochs_bfgs, architecture=architecture_monk3, neurons=neurons_monk3)

    param_bfgs_cup = GridSearchBFGSParams(c_1=c_1, c_2=c_2, theta=theta, regularization=regularization, epsilon=epsilon,
                                          epoch=epochs_bfgs, architecture=architecture_cup, neurons=neurons_cup)
    print "\nBFGS-MONK1\n"
    bfgs_training_monk(train_file=monk1_train_file, grid_search_param=param_bfgs_m1, save_dir=bfgs_dir + monk1)
    print "\nBFGS-MONK2\n"
    bfgs_training_monk(train_file=monk2_train_file, grid_search_param=param_bfgs_m2, save_dir=bfgs_dir + monk2)
    print "\nBFGS-MONK3\n"
    bfgs_training_monk(train_file=monk3_train_file, grid_search_param=param_bfgs_m3, save_dir=bfgs_dir + monk3)
    print "\nBFGS-CUP\n"
    tr_patterns, tr_targets, vl_patterns, vl_targets = read_cup_data()
    grid_search_BFGS(param_bfgs_cup, EuclideanError(), tr_patterns, tr_targets,
                      vl_patterns, vl_targets,
                      n_trials=5, save_in_dir=lbfgs_dir + cup)

if __name__ == '__main__':
    # architectures
    architecture_monk1 = [[17, 20, 10, 1]]
    architecture_monk2 = [[17, 20, 1]]
    architecture_monk3 = [[17, 20, 20, 1]]
    architecture_cup = [[10, 30, 20, 10, 2], [10, 40, 20, 2], [10, 100, 50, 2], [10, 100, 50, 25, 2], [10, 200, 100, 50, 2]]

    # neurons
    neurons_monk1 = [[InputNeuron, TanHNeuron, TanHNeuron, TanHNeuron]]
    neurons_monk2 = [[InputNeuron, TanHNeuron, TanHNeuron]]
    neurons_monk3 = [[InputNeuron, TanHNeuron, TanHNeuron, TanHNeuron]]
    neurons_cup = [[InputNeuron, TanHNeuron, TanHNeuron, TanHNeuron, OutputNeuron],
                   [InputNeuron, TanHNeuron, TanHNeuron, OutputNeuron],
                   [InputNeuron, TanHNeuron, TanHNeuron, OutputNeuron],
                   [InputNeuron, TanHNeuron, TanHNeuron, TanHNeuron, OutputNeuron],
                   [InputNeuron, TanHNeuron, TanHNeuron, TanHNeuron, OutputNeuron]]

    regularization = [0.05]

    #### SGD ####
    learning_rate = [0.01, 0.001]
    momentum = [0.9]
    batch_size = [16]
    epochs_sgd = 100

    #### BFGS / LBFGS ####
    c_1 = [0.0001, 0.001, 0.01]
    c_2 = [0.5, 0.7, 0.9]

    m_monk = [1, 5, 10, 15, 20]
    m_cup = [1, 10, 20, 30, 40, 50, 100, 200]

    epsilon = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    theta = [0.9]
    epochs_lbfgs = 100
    epochs_bfgs = 100

    if len(sys.argv) == 2:
        if sys.argv[1] == 'sgd':
            sgd_all_grid_search()
        elif sys.argv[1] == 'lbfgs':
            lbfgs_all_grid_search()
        elif sys.argv[1] == 'bfgs':
            bfgs_all_grid_search()
    else:
        print "usage:", sys.argv[0], "{sgd, bfgs, lbfgs}"
