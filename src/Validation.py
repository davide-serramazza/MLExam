import matplotlib.pyplot as plt
import copy
import numpy as np
from Neural_network import *
import pandas as pd

class grid_search_parameter:
    def __init__(self,learning_rate,momentum,batch_size,architecture,neurons, regolarization, epoch):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.architecture = architecture
        self.neurons = neurons
        self.regolarization = regolarization
        self.epoch = epoch

def transf_value(value):
    """
    transform value of variable to suitable string
    :param value: string to be transformef
    :return:
    """
    return str(value).replace(".",",")

def tranf_arc(architecture):
    """
    transoform architecture value to suitable string
    :param architecture: architecture to be transformed
    :return:
    """
    string = "["
    for i in architecture:
        string += str(i)+","
    string +="]"
    return string

def grid_search(parameter, loss_obj, tr_patterns,tr_labels,vl_patterns,vl_labels, n_trials):
    """
    grid search for optimal hyperparameter
    :param network: network to be trained
    :param loss_obj: loss used
    :param reguaritazion:
    :param n_trials: n of random trails for each value
    :param tr_patterns: traning set patterns
    :param tr_labels: traning set target
    :param vl_patterns: validation set patterns
    :param vl_labels: validation set target
    :return:
    """
    n_figure = 0  # index of figures
    # for every value
    for reg in parameter.regolarization:
        for lr in parameter.learning_rate:
            for mo in parameter.momentum:
                for bat in parameter.batch_size:
                    for arc,neur in zip(parameter.architecture,parameter.neurons):
                            # initialize lists for saving reslut
                            squared_error_avarage = np.zeros(parameter.epoch)
                            misClass_error_avarage = np.zeros(parameter.epoch)
                            squared_error_validation_avarage = np.zeros(parameter.epoch)
                            misClass_error_validation_avarage = np.zeros(parameter.epoch)
                            # n_trials then avarage
                            for n in range(n_trials):
                                # buid a new network
                                network = Network(arc,neur)
                                # train
                                squared_error,misClass_error, squared_error_validation,misClass_error_validation = \
                                    network.train(data=tr_patterns,
                                                  targets=tr_labels,
                                                  vl_data=vl_patterns,
                                                  vl_targets=vl_labels,
                                                  lossObject=loss_obj, epochs=parameter.epoch,
                                                  learning_rate=lr,
                                                  batch_size=bat, momentum=mo, regularization=0.01)

                                #append result of single epoch in list previously created
                                squared_error_avarage +=squared_error
                                misClass_error_avarage += misClass_error
                                squared_error_validation_avarage += squared_error_validation
                                misClass_error_validation_avarage += misClass_error_validation

                            # taking mean
                            squared_error_avarage/= (( float(n_trials)/2 *len(tr_patterns)))
                            # dividing by n_trials/2 beacuse our implementation of squared error is (target-output)/2
                            #dividing by len(tr_patterns) beacuse loss return absolute value, not mean
                            misClass_error_avarage/=( float(n_trials) *len(tr_patterns))
                            squared_error_validation_avarage/=( float(n_trials)/2 *len(vl_patterns))
                            misClass_error_validation_avarage/=( float(n_trials) *len(vl_patterns))

                            # plot result
                            plt.figure(n_figure, dpi=300) # select figure number 'n_figure'
                            plt.subplot(2, 1, 1)
                            plt.plot(range(1, len(misClass_error_avarage) + 1), misClass_error_avarage, '--')
                            plt.plot(range(1, len(misClass_error_validation_avarage) + 1), misClass_error_validation_avarage, '-')
                            plt.legend(['training set', 'validation set'])
                            plt.xlabel("epochs")
                            plt.ylabel("misclassification")
                            #plot squaredError
                            plt.subplot(2, 1, 2)
                            plt.plot(range(1, len(squared_error_avarage) + 1), squared_error_avarage, '--')
                            plt.plot(range(1, len(squared_error_validation_avarage) + 1),squared_error_validation_avarage, '-')
                            plt.legend(['training set', 'validation set'])
                            plt.xlabel("epochs")
                            plt.ylabel("squared error")
                            s = "../image/lr_"+transf_value(lr)+"-mo_"+transf_value(mo)+"-bat:"+transf_value(bat)+"-arc_"+tranf_arc(arc)
                            plt.tight_layout()  # minimize overlap of subplots
                            plt.savefig(s)
                            n_figure += 1 # increment to create a new figure
                            plt.close()


def holdout(frac, train_data):
    """
    Splits train_data in training set and validation set. Each set has the property that
    the positive and negative example follow roughly the same distribution of the original set.
    The training set is 'frac' percent of the data whereas the validation set is the remaining
    (1-frac) percent.

    :param frac: fraction of the dataset used for training
    :param train_data: dataset
    :return: training_set and validation_set
    """
    # shuffle data set
    train_data = train_data.reindex(np.random.permutation(train_data.index))
    # devide in positive and negative examples
    positive_set = train_data[train_data["label"] == 1]  # len = 64
    negative_set = train_data[train_data["label"] == 0]  # len = 105

    # compute length of partitions given frac
    len_pos_training = int(np.round(frac * len(positive_set)))  # for training set
    len_neg_training = int(np.round(frac * len(negative_set)))  # for validation set
    len_pos_validation = len(positive_set) - len_pos_training        # for training set
    len_neg_validation = len(negative_set) - len_neg_training        # for validation set

    positive_set_partition = positive_set.head(len_pos_training)  # len 45
    negative_set_partition = negative_set.head(len_neg_training)  # len 74
    positive_set_other = positive_set.head(len_pos_validation)  # len 19
    negative_set_other = negative_set.head(len_neg_validation)  # len 31
    training_set = pd.concat([positive_set_partition, negative_set_partition])  # len 119
    validation_set = pd.concat([positive_set_other, negative_set_other])  # len 50
    return training_set, validation_set
