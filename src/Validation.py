import matplotlib.pyplot as plt
import copy
import numpy as np

def grid_search(network,loss_obj,reguaritazion,n_trials, tr_patterns,tr_labels,vl_patterns,vl_labels):
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
    # for every value (per adesso solo numero di iterazione)
    for val in reguaritazion:
        # initialize lists for saving reslut
        squared_error_avarage = np.array([])
        misClass_error_avarage = np.array([])
        squared_error_validation_avarage = np.array([])
        misClass_error_validation_avarage = np.array([])
        # 10 trails then avarage
        for n in range(n_trials):
            # reinitilize weigths
            network.intialize_weight()
            # train
            squared_error,misClass_error, squared_error_validation,misClass_error_validation = network.train(
                data=tr_patterns,targets=tr_labels, vl_data=vl_patterns, vl_targets=vl_labels , lossObject=loss_obj,
                                epochs=200, learning_rate=val, batch_size=1, momentum=0.0, regularization=0.01)

            #append result of single epoch in list previously created
            if n==0:
                # in n==0 create list
                squared_error_avarage = copy.deepcopy( squared_error)
                misClass_error_avarage = copy.deepcopy( misClass_error)
                squared_error_validation_avarage = copy.deepcopy( squared_error_validation)
                misClass_error_validation_avarage =copy.deepcopy( misClass_error_validation)
            else:
                #else sum
                squared_error_avarage +=squared_error
                misClass_error_avarage += misClass_error
                squared_error_validation_avarage += squared_error_validation
                misClass_error_validation_avarage += misClass_error_validation

        # taking mean
        squared_error_avarage/= (( float(n_trials)/2 *len(tr_patterns)))
        # dividing by n_trials/2 beacuse our implementation of squared error is 2(output-traning)
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
        plt.show()


def hold_out(network,loss_obj,pattrns,targets,frac):
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
    grid_search(network,loss_obj,[0.02,0.03],5, tr_pattern,tr_labels,vl_pattern,vl_labels)