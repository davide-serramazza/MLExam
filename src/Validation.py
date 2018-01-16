import matplotlib.pyplot as plt
import copy
import numpy as np
from Neural_network import *

class grid_search_parameter:
    def __init__(self,learning_rate,momentum,regularization,architecture,neurons):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization
        self.architecture = architecture
        self.neurons = neurons

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
        string += str(i)
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
    fixed_number_epoch = 3
    # for every value
    for lr in parameter.learning_rate:
        for mo in parameter.momentum:
            for reg in parameter.regularization:
                for arc,neur in zip(parameter.architecture,parameter.neurons):
                    # initialize lists for saving reslut
                    squared_error_avarage = np.zeros(fixed_number_epoch)
                    misClass_error_avarage = np.zeros(fixed_number_epoch)
                    squared_error_validation_avarage = np.zeros(fixed_number_epoch)
                    misClass_error_validation_avarage = np.zeros(fixed_number_epoch)
                    # n_trials then avarage
                    for n in range(n_trials):
                        # buid a new network
                        network = Network(arc,neur)
                        # train
                        squared_error,misClass_error, squared_error_validation,misClass_error_validation = network.train(
                            #specify dataset and loss
                    data=tr_patterns,targets=tr_labels, vl_data=vl_patterns, vl_targets=vl_labels , lossObject=loss_obj,
                            #specify hyperparameters
            epochs=fixed_number_epoch, learning_rate=lr, batch_size=len(tr_patterns), momentum=mo, regularization=reg)

                        #append result of single epoch in list previously created
                        squared_error_avarage +=squared_error
                        misClass_error_avarage += misClass_error
                        squared_error_validation_avarage += squared_error_validation
                        misClass_error_validation_avarage += misClass_error_validation

                    # taking mean
                    squared_error_avarage/= (( float(n_trials)/2 *len(tr_patterns)))
                    # dividing by n_trials/2 beacuse our implementation of squared error is (target-output)/2
                    #dividing by len(tr_patterns) beacuse loss return absolute value, not mean
                    misClass_error_avarage/=( n_trials *len(tr_patterns))
                    squared_error_validation_avarage/=( float(n_trials)/2 *len(vl_patterns))
                    misClass_error_validation_avarage/=( n_trials *len(vl_patterns))

                    # plot result
                    plt.subplot(1, 2, 1)
                    plt.plot(range(len(misClass_error_avarage)), misClass_error_avarage)
                    plt.plot(range(len(misClass_error_validation_avarage)), misClass_error_validation_avarage)
                    plt.legend(['traing set', 'validation set'])
                    plt.xlabel("epochs")
                    plt.ylabel("misClassification")
                    #plot squaredError
                    plt.subplot(1,2,2)
                    plt.plot(range(len(squared_error_avarage)),squared_error_avarage)
                    plt.plot(range(len(squared_error_validation_avarage)),squared_error_validation_avarage)
                    plt.legend(['traing set', 'validation set'])
                    plt.xlabel("epochs")
                    plt.ylabel("squaredError")
                    s = "../image/lr_"+transf_value(lr)+" mo_"+transf_value(mo)+" reg:"+transf_value(reg)+" arc_"+tranf_arc(arc)
                    plt.savefig(s)


def hold_out(pattrns,targets,frac):
    """
    hold out function: divide dataset in traning and validation then call grid search
    :param network: network to be train
    :param loss_obj: loss used in traning
    :param pattrns: dataset patterns
    :param targets: dataset targets
    :param frac: fraction of data set for traning set (fraction of validation is 1-frac)
    :return:
    """
    lenght = int (len(pattrns)*frac )
    tr_pattern = pattrns[:lenght]
    tr_labels = targets[:lenght]
    vl_pattern = pattrns[lenght:]
    vl_labels = targets[lenght:]
    return tr_pattern,tr_labels,vl_pattern,vl_labels