import pandas as pd
import numpy as np


def holdout(frac, train_data):
    """
    Splits the dataframe 'train_data' in training set and validation set. Each set has the property that
    the positive and negative example follow roughly the same distribution of the original set.
    The training set is 'frac' percent of the data whereas the validation set is the remaining
    (1-frac) percent.

    :param frac: fraction of the dataset used for training
    :param train_data: dataset
    :return: training_set and validation_set
    """
    # shuffle data set
    train_data = train_data.reindex(np.random.permutation(train_data.index))
    # divide in positive and negative examples
    positive_set = train_data[train_data["label"] == 1]
    negative_set = train_data[train_data["label"] == 0]

    # compute length of partitions given frac
    len_pos_training = int(np.round(frac * len(positive_set)))  # for training set
    len_neg_training = int(np.round(frac * len(negative_set)))  # for validation set
    len_pos_validation = len(positive_set) - len_pos_training   # for training set
    len_neg_validation = len(negative_set) - len_neg_training   # for validation set

    positive_set_partition = positive_set.head(len_pos_training)
    negative_set_partition = negative_set.head(len_neg_training)

    positive_set_other = positive_set.tail(len_pos_validation)
    negative_set_other = negative_set.tail(len_neg_validation)

    training_set = pd.concat([positive_set_partition, negative_set_partition])
    validation_set = pd.concat([positive_set_other, negative_set_other])

    # shuffle training and validation set
    training_set = training_set.reindex(np.random.permutation(training_set.index))
    validation_set = validation_set.reindex(np.random.permutation(validation_set.index))
    return training_set, validation_set

#### PREPROCESSING ML CUP DATASET #####

def divide_patterns_labels(partition, feature_col, target_col):
    patterns = partition[feature_col].values
    labels = partition[target_col].values
    return patterns, labels


def holdout_cup(patterns, labels, fraction_tr):
    # calculate size
    len_partion = int(fraction_tr * len(patterns))

    # divide train/set
    first_partition_patterns = patterns[:len_partion]
    first_partition_labels = labels[:len_partion]
    second_partition_pattens = patterns[len_partion:]
    second_partition_labels = labels[len_partion:]
    return first_partition_patterns, first_partition_labels, second_partition_pattens, second_partition_labels
