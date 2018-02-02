import pandas as pd
from Neural_network import *
import matplotlib.pyplot as plt
from Validation import *
from sklearn.preprocessing import *
import time
from Validation_CM import *


def main():
    # read dataset
    df = pd.read_csv("../MLCup/ML-CUP17-TR_shuffled.csv", comment='#')
    features_col = ["input1","input2","input3","input4","input5","input6","input7", "input8","input9","input10"]
    targets_col = ["target_x", "target_y"]

    # divide pattern and targets
    patterns,labels = divide_patterns_labels(df,features_col,targets_col)

    # normalization objects used to normalize features (only the features!)
    normalizer = MinMaxScaler(feature_range=(-1, 1))
    x_scaled = normalizer.fit_transform(patterns)
    y_scaled = normalizer.fit_transform(labels)

    # divide in tr,vl and ts set
    first_partition_patterns, first_partition_labels, test_patterns, test_targets = holdout_cup(patterns,
                                                                                                labels, 0.8)
    tr_patterns, tr_targets, vl_patterns, vl_targets = holdout_cup(first_partition_patterns
                                                                   ,first_partition_labels, 0.8)

    architecture = [[10,10,2]]
    neurons = [[InputNeuron,SigmoidNeuron,OutputNeuron]]
    epochs = 200
    learning_rate = [0.2]
    batch_size = [256]
    momentum = [0.5]
    regularization = [0.01]
    parameter = grid_search_parameter(learning_rate, momentum, batch_size, architecture, neurons, regularization, epochs)
    # create loss
    loss_obj = EuclideanError(normalizer=None)

    start_time = time.time()
    grid_search(parameter, loss_obj, tr_patterns, tr_targets, vl_patterns, vl_targets,
                n_trials=5, save_in_dir="../image/")

    elapsed_time = time.time() - start_time
    print "time in grid search:", elapsed_time

    """
    architecture = [10,10,2]
    neurons = [InputNeuron,SigmoidNeuron,OutputNeuron]
    epochs = 200
    theta=[0.9]
    c_1=[0.0001]
    c_2=[0.9]
    regularization = [0.01]
    parameter = grid_search_CM_parameter(c_1,c_2,theta,regularization,epochs,architecture,neurons)
    # create loss
    loss_obj = EuclideanError(normalizer=None)

    start_time = time.time()
    grid_search_CM(parameter, loss_obj, tr_patterns, tr_targets, vl_patterns, vl_targets,
                n_trials=1, save_in_dir="../image/MLCup_CM/")

    elapsed_time = time.time() - start_time
    print "time in grid search:", elapsed_time

"""

def divide_patterns_labels(partition, feature_col, target_col):
    patterns = partition[feature_col].values
    labels = partition[target_col].values
    return patterns, labels


def holdout_cup(patterns, labels, frac_tr):
    # calculate size
    len_partion = int(frac_tr * len(patterns))

    first_partition_patterns = patterns[:len_partion]
    first_partition_labels = labels[:len_partion]
    second_partition_pattens = patterns[len_partion:]
    second_partition_labels = labels[len_partion:]
    return first_partition_patterns, first_partition_labels, second_partition_pattens, second_partition_labels


if __name__ == "__main__":
    main()
