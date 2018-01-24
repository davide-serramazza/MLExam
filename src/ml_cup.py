import pandas as pd
from Neural_network import *
import matplotlib.pyplot as plt
from Validation import *
import time


def main():
    # read dataset
    df = pd.read_csv("../MLCup/ML-CUP17-TR.csv", comment='#', header=None)
    features_col = ["input1","input2","input3","input4","input5","input6","input7", "input8","input9","input10"]
    targets_col = ["target_x", "target_y"]
    df.columns = ["id"] + features_col + targets_col

    # shuffle dataset and holdout
    frac_tr, frac_vl, frac_ts = 0.5, 0.25, 0.25
    first_partition, test_data = holdout_cup(df,0.75)
    traning_data, validation_data = holdout_cup(first_partition,0.75)

    # divide patterns and targets
    tr_patterns, tr_targets = divide_patterns_labels(traning_data,features_col,targets_col)
    vl_patterns, vl_targets = divide_patterns_labels(validation_data,features_col,targets_col)
    te_patterns, te_targets = divide_patterns_labels(test_data,features_col,targets_col)

    #create network
    learning_rate = [0.001 , 0.005, 0.01]
    momentum = [0.0]
    batch_size = [1]
    architecture = [ [10,10,2], [10,10,5,2], [10,10,5,5,2] ]
    neurons = [ [InputNeuron,TanHNeuron,OutputNeuron], [InputNeuron,TanHNeuron, TanHNeuron, OutputNeuron],
                [InputNeuron, TanHNeuron,TanHNeuron,TanHNeuron, OutputNeuron]]
    regularization = [0.01,0.05]
    epochs = 300
    parameter = grid_search_parameter(learning_rate,momentum,batch_size,architecture,neurons,regularization,epochs)
    # create loss
    loss_obj = SquaredError("tangentH")
    start_time = time.time()
    grid_search(parameter,loss_obj,tr_patterns,tr_targets,vl_patterns,vl_targets,5)
    elapsed_time = time.time() - start_time
    print elapsed_time

def divide_patterns_labels(partion,feature_col, target_col) :
    patterns = partion[feature_col].values
    labels = partion[target_col].values
    return patterns,labels

def holdout_cup(dataset, frac_tr):
    df = dataset.reindex(np.random.permutation(dataset.index))
    len_partion = int(frac_tr * len(df))
    first_partition = df[:len_partion]
    second_partition = df[len_partion:]
    return first_partition, second_partition


if __name__ == "__main__":
    main()
