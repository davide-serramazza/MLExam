import pandas as pd
from monk_benchmark import *

def main():
    columns = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'id']
    # 1. traning set
    train_file = "../monk_datasets/monks-2.train"
    train_data = pd.read_csv(train_file, delim_whitespace=True, header=None)
    train_data.columns = columns

    # 2. test set
    test_file = "../monk_datasets/monks-2.test"
    test_data = pd.read_csv(test_file, delim_whitespace=True, header=None)
    test_data.columns = columns
    #getting patterns and labels
    encoding = [3, 3, 2, 3, 4, 2]
    training_patterns, test_patterns = decode_patterns(encoding,['f1', 'f2', 'f3', 'f4', 'f5', 'f6'],
                                                             train_data,test_data)
    training_labels,test_labels = transform_labels(train_data,test_data)
    architecture = [17,10,1]
    neurons = [InputNeuron,TanHNeuron,TanHNeuron]
    network = Network(architecture,neurons)
    lossObject = SquaredError("tangentH")

    epochs = 25
    learning_rate = 0.2
    batch_size = 10
    momentum = 0.6
    regularization = 0

    squared_error,misClass_error, squared_error_test,misClass_error_test = network.train(
        data=training_patterns,targets=training_labels,eval_data=test_patterns,eval_targets=test_labels,
            lossObject = lossObject,epochs=epochs, learning_rate=learning_rate , batch_size=batch_size , momentum=momentum,
    regularization=regularization)
    # getting avarage
    squared_error /= float(2*len(training_patterns))
    squared_error_test /= float (2*len(test_patterns))
    misClass_error /= float (len(training_patterns))
    misClass_error_test /= float(len(test_patterns))
    print_result(misClass_error , misClass_error_test , squared_error , squared_error_test , "test set",
                 architecture,batch_size, learning_rate,momentum,regularization,1)
    print "accuracy", 1-misClass_error[-1]
    print "accuracy test", 1-misClass_error_test[-1]
    print "squared error", squared_error[-1]
    print "squared error test", squared_error_test[-1]





if __name__ == '__main__':
    main()