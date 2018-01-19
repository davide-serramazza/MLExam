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

    network.train(data=training_patterns,targets=training_labels,eval_data=test_patterns,eval_targets=test_labels,
                  lossObject = lossObject,epochs=1, learning_rate=0.1,batch_size=1,momentum=0.0

                  )




if __name__ == '__main__':
    main()