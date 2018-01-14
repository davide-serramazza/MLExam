import matplotlib.pyplot as plt
import copy
import numpy as np

def grid_search(network,loss_obj,n_epoch,tr_patterns,tr_labels,vl_pattern,vl_labels):
    # for every value (per adesso solo numero di iterazione)
    for val in n_epoch:
        # create net's copy to train
        net = copy.deepcopy(network)
        # create list used for append each epoch result
        squared_error = []
        misClass_error = []
        squared_error_validation = []
        misClass_error_validation = []
        for n in range(val):
            # train
            squared_error_epoch,misClass_epoch = net.train(data=tr_patterns, targets=tr_labels,lossObject=loss_obj, epochs=1, learning_rate=0.02,
                      batch_size=1, momentum=0.0, regularization=0.01)
            #error on validation set
            squared_error_validation_epoch,misClass_error_validation_epoch = net.validation_error(vl_pattern,
                                                                                                  vl_labels,loss_obj)
            #append result of single epoch in an array
            squared_error.append(squared_error_epoch)
            misClass_error.append(misClass_epoch)
            squared_error_validation.append(squared_error_validation_epoch)
            misClass_error_validation.append(misClass_error_validation_epoch)
        # plot result
        plt.subplot(1, 2, 1)
        plt.plot(range(len(misClass_error)), misClass_error)
        plt.plot(range(len(misClass_error_validation)), misClass_error_validation)
        plt.legend(['traing set', 'validation set'])
        plt.xlabel("epochs")
        plt.ylabel("misClassification")
        #plt.show()
        #plot squaredError
        plt.subplot(1,2,2)
        squareE = np.asarray(squared_error) * 2 / len(squared_error)  # TODO: divide by len(losses) to obtain MSE?
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
    # grid search
    grid_search(network,loss_obj,[100], tr_pattern,tr_labels,vl_pattern,vl_labels)