from sklearn.preprocessing import *
import time
from Validation import *
from grid_search import *
from utils import *
from Neural_network import *


def main():

    # 1. read dataset
    df = pd.read_csv("../MLCup/ML-CUP17-TR_shuffled.csv", comment='#')
    features_col = ["input1","input2","input3","input4","input5","input6","input7", "input8","input9","input10"]
    targets_col = ["target_x", "target_y"]

    # 2. divide pattern and targets
    patterns, labels = divide_patterns_labels(df,features_col,targets_col)

    # 3. divide in tr,vl and ts set
    first_partition_patterns, first_partition_labels, test_patterns, test_targets = holdout_cup(patterns,
                                                                                                labels, 0.8)
    tr_patterns, tr_targets, vl_patterns, vl_targets = holdout_cup(first_partition_patterns
                                                                   ,first_partition_labels, 0.8)
    # 4. define architecture and hyperparameters
    architecture = [[10,15,2], [10,20,2]]
    neurons = [[InputNeuron,SigmoidNeuron,OutputNeuron], [InputNeuron,SigmoidNeuron,OutputNeuron]]
    epochs = 100
    theta = [0.9]
    c_1 = [0.0001]
    c_2 = [0.8]
    regularization = [0.00001, 0.0001]
    m = [30, 50]
    parameter = GridSearchBFGSParams(c_1=c_1, c_2=c_2, theta=theta, regularization=regularization,
                                     epoch=epochs, architecture=architecture, neurons=neurons, epsilon=[0.001])

    loss_obj = EuclideanError(normalizer=None)

    # 6. train
    start_time = time.time()
    grid_search_BFGS(parameter, loss_obj, tr_patterns, tr_targets, vl_patterns, vl_targets,
                     n_trials=3, save_in_dir="../grid_search_results/bfgs/cup/a")

    elapsed_time = time.time() - start_time
    print "time in grid search:", elapsed_time

if __name__ == "__main__":
    main()
