import pandas as pd
from Neural_network import *
import matplotlib.pyplot as plt
from Validation import *
from sklearn.preprocessing import *
import time


def main():
    # read dataset
    df = pd.read_csv("../MLCup/ML-CUP17-TR.csv", comment='#', header=None)
    features_col = ["input1","input2","input3","input4","input5","input6","input7", "input8","input9","input10"]
    targets_col = ["target_x", "target_y"]
    df.columns = ["id"] + features_col + targets_col

    # divide pattern and targets
    pattern,labels = divide_patterns_labels(df,features_col,targets_col)

    # normalization objects used to normalize features (only the features!)
    normalizer = MinMaxScaler(feature_range=(-1, 1))
    x_scaled = normalizer.fit_transform(pattern)
    y_scaled = normalizer.fit_transform(labels)

    # divide in tr,vl and ts set
    first_partition_patterns, first_partition_labels, test_patterns, test_targets = holdout_cup(x_scaled,y_scaled, 0.9)
    tr_patterns, tr_targets, vl_patterns, vl_targets = holdout_cup(first_partition_patterns,first_partition_labels, 0.9)

    # create network
    learning_rate = [0.01] #[0.05, 0.1]
    momentum = [0.5] #[0.25, 0.5]
    batch_size = [64]#[10]
    architecture = [[10,20,20,2]] #[ [10,20,20,2], [10,20,15,10,2]]
    neurons = [[InputNeuron,TanHNeuron,TanHNeuron,OutputNeuron]]#[[InputNeuron,TanHNeuron,TanHNeuron,OutputNeuron],
               # [InputNeuron,TanHNeuron,TanHNeuron, TanHNeuron, OutputNeuron]]
    regularization = [0.01]
    epochs = 300
    parameter = grid_search_parameter(learning_rate, momentum, batch_size, architecture, neurons, regularization, epochs)
    # create loss
    loss_obj = EuclideanError(normalizer)

    start_time = time.time()
    grid_search(parameter, loss_obj, tr_patterns, tr_targets, vl_patterns, vl_targets,
                n_trials=5, save_in_dir="../image/MLCup")
    elapsed_time = time.time() - start_time
    print "time in grid search:", elapsed_time


def divide_patterns_labels(partition, feature_col, target_col):
    patterns = partition[feature_col].values
    labels = partition[target_col].values
    return patterns, labels


def holdout_cup(patterns, labels, frac_tr):
    # shuffle dataset
    permumation = np.random.permutation(len(patterns))
    # calculate size
    len_partion = int(frac_tr * len(patterns))

    first_partition_patterns = patterns[:len_partion]
    first_partition_labels = labels[:len_partion]
    second_partition_pattens = patterns[len_partion:]
    second_partition_labels = labels[len_partion:]
    return first_partition_patterns, first_partition_labels, second_partition_pattens, second_partition_labels


if __name__ == "__main__":
    main()
