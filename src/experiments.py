from Validation import *
from grid_search import *
from Neural_network import *
import sys

# files training set
monk1_train_file = "../monk_datasets/monks-1.train"
monk2_train_file = "../monk_datasets/monks-2.train"
monk3_train_file = "../monk_datasets/monks-3.train"
cup_train_file = "../MLCup/ML-CUP17-TR_shuffled.csv"

# save experiments in
sgd_dir = "../grid_search_results/sgd/"
bfgs_dir = "../grid_search_results/bfgs/"
lbfgs_dir = "../grid_search_results/lbfgs/"

# subdirectories to save experiements
monk1 = "monk1/"
monk2 = "monk2/"
monk3 = "monk3/"
cup = "cup/"

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
    param = GridSearchSGDParams(learning_rate=learning_rate,
                                momentum=momentum,
                                batch_size=batch_size,
                                architecture=architecture,
                                neurons=neurons,
                                regularization=no_regularization,
                                epoch=epochs_sgd)
    param_reg = GridSearchSGDParams(learning_rate, momentum, batch_size, architecture, neurons,
                                    regularization=regularization,
                                    epoch=epochs_sgd)
    param_cup = GridSearchSGDParams(learning_rate, momentum, batch_size,
                                    architecture=architecture_cup,
                                    neurons=neurons_cup,
                                    regularization=regularization, epoch=epochs_sgd)
    #### SGD - MONK 1 ###
    print "\nSGD-MONK1\n"
    sgd_training_monk(train_file=monk1_train_file, grid_search_param=param, save_dir=sgd_dir + monk1)
    #### SGD - MONK 2 ###
    print "\nSGD-MONK2\n"
    sgd_training_monk(train_file=monk2_train_file, grid_search_param=param, save_dir=sgd_dir + monk2)
    #### SGD - MONK 3 ###
    print "\nSGD-MONK3\n"
    sgd_training_monk(train_file=monk3_train_file, grid_search_param=param_reg, save_dir=sgd_dir + monk3)
    #### SGD - CUP ###
    print "\nSGD-CUP\n"
    # 1. read dataset
    tr_patterns, tr_targets, vl_patterns, vl_targets = read_cup_data()
    grid_search_SGD(param_cup, EuclideanError(), tr_patterns, tr_targets,
                    vl_patterns, vl_targets, n_trials=5, save_in_dir=sgd_dir + cup)


def lbfgs_all_grid_search():
    global tr_patterns, tr_targets, vl_patterns, vl_targets

    param_lbfgs_m1 = GridSearchLBFGSParams(c_1=c_1_monk1, c_2=c_2, theta=theta, regularization=no_regularization, m=m_monk,
                                        epoch=epochs_lbfgs, architecture=architecture, neurons=neurons)

    param_lbfgs = GridSearchLBFGSParams(c_1=c_1_monk23, c_2=c_2, theta=theta, regularization=no_regularization, m=m_monk,
                                        epoch=epochs_lbfgs, architecture=architecture, neurons=neurons)

    param_lbfgs_reg = GridSearchLBFGSParams(c_1=c_1_monk23, c_2=c_2, theta=theta, regularization=regularization, m=m_monk,
                                            epoch=epochs_lbfgs, architecture=architecture, neurons=neurons)
    param_lbfgs_cup = GridSearchLBFGSParams(c_1=c_1_cup, c_2=c_2, theta=theta, regularization=regularization, m=m_cup,
                                            epoch=epochs_lbfgs, architecture=architecture_cup, neurons=neurons_cup)
    print "\nLBFGS-MONK1\n"
    #lbfgs_training_monk(train_file=monk1_train_file, grid_search_param=param_lbfgs_m1, save_dir=lbfgs_dir + monk1)
    print "\nLBFGS-MONK2\n"
    #lbfgs_training_monk(train_file=monk2_train_file, grid_search_param=param_lbfgs, save_dir=lbfgs_dir + monk2)
    print "\nLBFGS-MONK3\n"
    #lbfgs_training_monk(train_file=monk3_train_file, grid_search_param=param_lbfgs_reg, save_dir=lbfgs_dir + monk3)
    print "\nLBFGS-CUP\n"
    tr_patterns, tr_targets, vl_patterns, vl_targets = read_cup_data()
    grid_search_LBFGS(param_lbfgs_cup, EuclideanError(), tr_patterns, tr_targets,
                      vl_patterns, vl_targets,
                      n_trials=5, save_in_dir=lbfgs_dir + cup)


if __name__ == '__main__':
    architecture = [[17, 20, 1], [17, 10, 10, 1]]
    neurons = [[InputNeuron, TanHNeuron, TanHNeuron], [InputNeuron, TanHNeuron, TanHNeuron, TanHNeuron]]

    architecture_cup = [[10, 10, 10, 2], [10, 20, 20, 2]]
    neurons_cup = [[InputNeuron, TanHNeuron, TanHNeuron, OutputNeuron],
                   [InputNeuron, TanHNeuron, TanHNeuron, OutputNeuron]]

    #### SGD ####
    learning_rate = [0.01, 0.001]
    momentum = [0.5, 0.9]
    batch_size = [16, 32]
    no_regularization = [0.0]
    regularization = [0.0, 0.01, 0.05]
    epochs_sgd = 200

    #### LBFGS ####
    c_1_monk1 = [0.0001, 0.001, 0.01] # for the monk c_1 = 0.001 is fine, frangio usa c_1=0.01, il libro dice c_1=0.0001
    c_1_monk23 = [0.0001, 0.001]  # crashed monk2 because alpha=100
    c_1_cup = [0.01] # also did one grid search with c_1=0.0001

    c_2 = [0.9]
    m_monk = [1, 10, 20, 30, 40, 50]
    m_cup = [1, 10, 20, 30, 40, 50, 100, 200]
    theta = [0.9]
    epochs_lbfgs = 100

    if len(sys.argv) == 2:
        if sys.argv[1] == 'sgd':
            sgd_all_grid_search()
        elif sys.argv[1] == 'lbfgs':
            lbfgs_all_grid_search()
    else:
        print "usage:", sys.argv[0], "{sgd,lbfgs}"
