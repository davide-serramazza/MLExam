import pandas as pd
from Validation import  *
from monk_benchmark import *
from Neural_network import *
from Validation_CM import *
import time

def main():
    train_file = "../monk_datasets/monks-3.train"

    # 1. load dataset
    columns = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'id']
    train_data = pd.read_csv(train_file, delim_whitespace=True, header=None)
    train_data.columns = columns

    # 2. hold out
    frac = 0.7
    training_set, validation_set = holdout(frac, train_data)

    # 3. decode patterns and transform targets
    encoding = [3, 3, 2, 3, 4, 2]
    features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
    training_patterns, validation_patterns = decode_patterns(encoding, features, training_set, validation_set)
    training_labels, validation_labels = transform_labels(training_set, validation_set)

    arch = [17,10,1]
    neuronsType = [InputNeuron, TanHNeuron, TanHNeuron]
    network = Network(arch, neuronsType)

    lossObject = SquaredError("tangentH")

    theta=[0.9]
    c_1=[0.0001,0.001,0.005]
    c_2=[0.9,0.85,0.5]
    regularizarion = [0.001]
    lossObject=lossObject
    epochs=50

    parameter = grid_search_CM_parameter(c_1,c_2,theta,regularizarion,epochs,arch,neuronsType)
    start_time = time.time()
    grid_search_CM(parameter,lossObject,training_patterns,training_labels,validation_patterns,validation_labels,5,"../image/monk3-reg/")
    elapsed_time = time.time() - start_time
    print "time in grid search:", elapsed_time


"""



    ##' esempio
    arch = [2, 2, 2]
    neuronsType = [InputNeuron, SigmoidNeuron, OutputNeuron]
    network = Network(arch, neuronsType)
    network.layers[1].neurons[0].weights = np.asarray([0.15, 0.2, 0.35])
    network.layers[1].neurons[1].weights = np.asarray([0.25, 0.3, 0.35])
    network.layers[2].neurons[0].weights = np.asarray([0.4, 0.45, 0.6])
    network.layers[2].neurons[1].weights = np.asarray([0.5, 0.55, 0.6])

    data = [[0.05, 0.1]]
    target = [[0.01, 0.99]]

   # network.trainBFGS(data,target,data,target,lossObject, 2)

    predictions = network.predict(data)
  #  print predictions

    network = Network(arch, neuronsType)
    network.layers[1].neurons[0].weights = np.asarray([0.15, 0.2, 0.35])
    network.layers[1].neurons[1].weights = np.asarray([0.25, 0.3, 0.35])
    network.layers[2].neurons[0].weights = np.asarray([0.4, 0.45, 0.6])
    network.layers[2].neurons[1].weights = np.asarray([0.5, 0.55, 0.6])
    network.trainBFGS(data,target,data,target,lossObject,0)
    predictions = network.predict(data)

    network = Network(arch, neuronsType)
    network.layers[1].neurons[0].weights = np.asarray([0.15, 0.2, 0.35])
    network.layers[1].neurons[1].weights = np.asarray([0.25, 0.3, 0.35])
    network.layers[2].neurons[0].weights = np.asarray([0.4, 0.45, 0.6])
    network.layers[2].neurons[1].weights = np.asarray([0.5, 0.55, 0.6])

"""

   # print predictions


### END EXAMPLE

if __name__ == '__main__':
    main()