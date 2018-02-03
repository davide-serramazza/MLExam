from monk_benchmark import *
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

    # 4. define architecture and hyper parameter
    arch = [17,10,1]
    neuronsType = [InputNeuron, TanHNeuron, TanHNeuron]
    lossObject = SquaredError("tangentH")

    theta=[0.9]
    c_1=[0.0001,0.001,0.005]
    c_2=[0.9,0.85,0.5]
    regularizarion = [0.001]
    lossObject=lossObject
    epochs=50

    # train
    parameter = grid_search_CM_parameter(c_1,c_2,theta,regularizarion,epochs,arch,neuronsType)
    start_time = time.time()
    grid_search_CM(parameter,lossObject,training_patterns,training_labels,validation_patterns,validation_labels,5,"../image/monk3-reg/")
    elapsed_time = time.time() - start_time
    print "time in grid search:", elapsed_time


if __name__ == '__main__':
    main()