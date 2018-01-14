import matplotlib.pyplot as plt
import copy
import numpy as np

def grid_search(network,loss_obj,n_epoch,tr_patterns,tr_labels,vl_pattern,vl_labels):
    # for every value (per adesso solo numero di iterazione)
    for val in n_epoch:
        # create lists for saving reslut
        squared_error_avarage = np.array([])
        misClass_error_avarage = np.array([])
        squared_error_validation_avarage = np.array([])
        misClass_error_validation_avarage = np.array([])
        # 10 trails then avarage
        for n in range(10):
            # create net's copy to train
            #TODO reinizialize the weights
            net = copy.deepcopy(network)
            # train
            squared_error,misClass_error, squared_error_validation,misClass_error_validation = net.train(data=tr_patterns,
                targets=tr_labels, vl_data=vl_pattern, vl_targets=vl_labels , lossObject=loss_obj,
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
        # taking mean
        squared_error_avarage/=10
        misClass_error_avarage/=10
        squared_error_validation_avarage/=10
        misClass_error_validation_avarage/=10

        # plot result
        plt.subplot(1, 2, 1)
        plt.plot(range(len(misClass_error)), misClass_error)
        plt.plot(range(len(misClass_error_validation)), misClass_error_validation)
        plt.legend(['traing set', 'validation set'])
        plt.xlabel("epochs")
        plt.ylabel("misClassification")
        #plot squaredError
        plt.subplot(1,2,2)
        plt.plot(range(len(squared_error)),squared_error)
        plt.plot(range(len(squared_error_validation)),squared_error_validation)
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
    grid_search(network,loss_obj,[50], tr_pattern,tr_labels,vl_pattern,vl_labels)