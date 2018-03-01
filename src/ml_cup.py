from sklearn.preprocessing import *
import time
from Validation_CM import *


def main():

    # 1. read dataset
    df = pd.read_csv("../MLCup/ML-CUP17-TR_shuffled.csv", comment='#')
    features_col = ["input1","input2","input3","input4","input5","input6","input7", "input8","input9","input10"]
    targets_col = ["target_x", "target_y"]

    # 2. divide pattern and targets
    patterns,labels = divide_patterns_labels(df,features_col,targets_col)

    # 3. normalization
    normalizer = MinMaxScaler(feature_range=(-1, 1))
    x_scaled = normalizer.fit_transform(patterns)
    y_scaled = normalizer.fit_transform(labels)

    # 4. divide in tr,vl and ts set
    first_partition_patterns, first_partition_labels, test_patterns, test_targets = holdout_cup(patterns,
                                                                                                labels, 0.8)
    tr_patterns, tr_targets, vl_patterns, vl_targets = holdout_cup(first_partition_patterns
                                                                   ,first_partition_labels, 0.8)
    # 5. define architecture and hyperparameters
    architecture = [10,10,2]
    neurons = [InputNeuron,SigmoidNeuron,OutputNeuron]
    epochs = 200
    theta=[0.9]
    c_1=[0.0001]
    c_2=[0.9]
    regularization = [0.01]
    parameter = grid_search_CM_parameter(c_1,c_2,theta,regularization,epochs,architecture,neurons)

    loss_obj = EuclideanError(normalizer=None)

    # 6. train
    start_time = time.time()
    grid_search_CM(parameter, loss_obj, tr_patterns, tr_targets, vl_patterns, vl_targets,
                   n_trials=1, save_in_dir="../image/MLCup_CM/")

    elapsed_time = time.time() - start_time
    print "time in grid search:", elapsed_time


def divide_patterns_labels(partition, feature_col, target_col):
    patterns = partition[feature_col].values
    labels = partition[target_col].values
    return patterns, labels


def holdout_cup(patterns, labels, frac_tr):

    # calculate size
    len_partion = int(frac_tr * len(patterns))

    #divide train set
    first_partition_patterns = patterns[:len_partion]
    first_partition_labels = labels[:len_partion]
    second_partition_pattens = patterns[len_partion:]
    second_partition_labels = labels[len_partion:]
    return first_partition_patterns, first_partition_labels, second_partition_pattens, second_partition_labels


if __name__ == "__main__":
    main()
