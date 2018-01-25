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

    # normalization objects used to normalize features (only the features!)
    normalizer = MinMaxScaler(feature_range=(-1, 1))
    standardizer = StandardScaler()

    # shuffle dataset and holdout
    first_partition, test_data = holdout_cup(df, 0.9)
    traning_data, validation_data = holdout_cup(first_partition, 0.9)

    # divide patterns and targets
    tr_patterns, tr_targets = divide_patterns_labels(traning_data,features_col,targets_col)
    vl_patterns, vl_targets = divide_patterns_labels(validation_data,features_col,targets_col)
    te_patterns, te_targets = divide_patterns_labels(test_data,features_col,targets_col)

    # normalize
    #tr_patterns = standardizer.fit_transform(tr_patterns)
    #vl_patterns = standardizer.fit_transform(vl_patterns)
    tr_patterns = normalizer.fit_transform(tr_patterns)
    vl_patterns = normalizer.fit_transform(vl_patterns)

    # create network
    """
    Bigger networks are better than smaller networks because they can express more complex functions.
    The risk of overfitting is adressed by the regularization strength.
    The bigger the network, the bigger the regularizaion.
    """
    learning_rate = [0.1]#[0.01, 0.05, 0.1, 0.25]
    momentum = [0.3]#[0, 0.2, 0.5]
    batch_size = [64]#[1, 64, 128, 256]
    architecture = [[10,20,20,2]]#[[10,50,2], [10,20,20,2], [10,20,20,20,2]]
    neurons = [[InputNeuron, TanHNeuron, TanHNeuron, OutputNeuron]]#[ [InputNeuron,TanHNeuron,OutputNeuron], [InputNeuron,TanHNeuron, TanHNeuron, OutputNeuron],
                #[InputNeuron, TanHNeuron,TanHNeuron,TanHNeuron, OutputNeuron]]
    regularization = [0.001]#[0.01, 0.05, 0.1]
    epochs = 100
    parameter = grid_search_parameter(learning_rate, momentum, batch_size, architecture, neurons, regularization, epochs)
    # create loss
    loss_obj = EuclideanError()

    start_time = time.time()
    grid_search(parameter, loss_obj, tr_patterns, tr_targets, vl_patterns, vl_targets, n_trials=5, save_in_dir="../image/MLCup/")
    elapsed_time = time.time() - start_time
    print "time in grid search:", elapsed_time

#### example
    #network = Network([10,10,10,2], [InputNeuron, LinearNeuron, LinearNeuron, OutputNeuron])
    #print network.predict(tr_patterns)[:3]
    #network.train(
    #    data=tr_patterns, targets=tr_targets, eval_data=vl_patterns, eval_targets=vl_targets,
    #    lossObject=loss_obj, epochs=epochs, learning_rate=learning_rate[0], batch_size=batch_size[0],
    #    momentum=momentum[0], regularization=regularization[0])
    #print network.predict(tr_patterns)[:3]
    #print tr_targets[:3]
###### end example


def divide_patterns_labels(partition, feature_col, target_col):
    patterns = partition[feature_col].values
    labels = partition[target_col].values
    return patterns, labels


def holdout_cup(dataset, frac_tr):
    # shuffle dataset
    df = dataset.reindex(np.random.permutation(dataset.index))
    len_partion = int(frac_tr * len(df))
    first_partition = df[:len_partion]
    second_partition = df[len_partion:]
    return first_partition, second_partition


if __name__ == "__main__":
    main()
