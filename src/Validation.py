import matplotlib.pyplot as plt
import copy
import numpy as np

def grid_search(network,loss_obj,n_epoch,n_trials, tr_patterns,tr_labels,vl_patterns,vl_labels):
    # for every value (per adesso solo numero di iterazione)
    for val in n_epoch:
        # create lists for saving reslut
        squared_error_avarage = np.array([])
        misClass_error_avarage = np.array([])
        squared_error_validation_avarage = np.array([])
        misClass_error_validation_avarage = np.array([])
        # 10 trails then avarage
        for n in range(n_trials):
            # create net's copy to train
            #TODO reinizialize the weights
            network.intialize_weight()
            # train
            squared_error,misClass_error, squared_error_validation,misClass_error_validation = network.train(
                data=tr_patterns,targets=tr_labels, vl_data=vl_patterns, vl_targets=vl_labels , lossObject=loss_obj,
                                epochs=val, learning_rate=0.03, batch_size=1, momentum=0.0, regularization=0.01)

            #append result of single epoch in list previously created
            if n==0:
                squared_error_avarage = copy.deepcopy( squared_error)
                misClass_error_avarage = copy.deepcopy( misClass_error)
                squared_error_validation_avarage = copy.deepcopy( squared_error_validation)
                misClass_error_validation_avarage =copy.deepcopy( misClass_error_validation)
            else:
                squared_error_avarage +=squared_error
                misClass_error_avarage += misClass_error
                squared_error_validation_avarage += squared_error_validation
                misClass_error_validation_avarage += misClass_error_validation

        # taking mean of dataset and 10 trials
        squared_error_avarage/= (( float(n_trials)/2 *len(tr_patterns)))
        # divide only by 5 beacuse our implementation of squared error is 2(output-traning)
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
    #divide dataset in traning e validation set
    lenght = int (len(pattrns)*frac )
    tr_pattern = pattrns[:lenght]
    tr_labels = targets[:lenght]
    vl_pattern = pattrns[lenght:]
    vl_labels = targets[lenght:]
    grid_search(network,loss_obj,[50],2, tr_pattern,tr_labels,vl_pattern,vl_labels)