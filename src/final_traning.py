from monk_benchmark import *
from time import time

def main():

    columns = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'id']
    # 1. traning set
    train_file = "../monk_datasets/monks-3.train"
    train_data = pd.read_csv(train_file, delim_whitespace=True, header=None)
    train_data.columns = columns

    # 2. test set
    test_file = "../monk_datasets/monks-3.test"
    test_data = pd.read_csv(test_file, delim_whitespace=True, header=None)
    test_data.columns = columns

    # 3. getting patterns and labels
    encoding = [3, 3, 2, 3, 4, 2]
    features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
    train_patterns, test_patterns = decode_patterns(encoding, features, train_data, test_data)
    train_labels, test_labels = transform_labels(train_data, test_data)

    # 4. define architecture and hyperparameters
    architecture = [17, 20, 20, 1]
    neurons = [InputNeuron, TanHNeuron, TanHNeuron, TanHNeuron]
    network = Network(architecture, neurons)
    loss_object = SquaredError("tangentH")
    epochs = 30
    learning_rate = 0.01
    batch_size = 16
    momentum = 0.7
    regularization = 0.05

    tic = time()
    # 5. train
    loss_train, misclass_train, loss_test, misclass_test = network.train(
            data=train_patterns, targets=train_labels, eval_data=test_patterns,eval_targets=test_labels,
            lossObject=loss_object, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
            momentum=momentum, regularization=regularization)
    toc = time()

    # 7. print
    print "accuracy train:", 1 - misclass_train[-1]
    print "accuracy test:", 1 - misclass_test[-1]
    print "squared error train:", loss_train[-1]
    print "squared error test:", loss_test[-1]
    print "training time:", (toc-tic)

    # 8. plot
    plot_train_test_learning_curve(loss_test, loss_train, misclass_test, misclass_train)


if __name__ == '__main__':
    main()