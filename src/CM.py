from monk_benchmark import *
from Validation import *
from Neural_network import *
from grid_search import *
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
    arch = [[17,10,1]]
    neuronsType = [[InputNeuron, TanHNeuron, TanHNeuron]]

    c_1 = [0.0001]
    c_2 = [0.9]
    theta = [0.9]
    reguralization = [0.001]
    m = [50]
    epochs = 100
    lossObject = SquaredError("tangentH")
    parameter = GridSearchBFGSParams(c_1,c_2,theta,reguralization,epochs,arch,neuronsType)
    # perform grid search
    grid_search_BFGS(parameter,lossObject,training_patterns,training_labels,validation_patterns,validation_labels,
                   n_trials=3,save_in_dir="../temp/new-")

    # try BFGS
    #tr_l, tr_ac, vl_l, vl_ac = network.trainBFGS(training_patterns, training_labels, validation_patterns, validation_labels,
    #                  theta=0.9, c_1=0.0001, c_2=0.9, lossObject=lossObject, regularization=0.01, epochs=10)
    #plt.plot(range(len(tr_l)), tr_l, "r", label="TR error")
    #plt.plot(range(len(vl_l)), vl_l, "b", label="VL error")
    #plt.show()

if __name__ == '__main__':
    main()