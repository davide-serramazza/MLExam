import pandas as pd
from Validation import  *
from monk_benchmark import *
from Neural_network import *

def main():
    train_file = "../monk_datasets/monks-1.train"

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

    lossObject = SquaredError("tangentH")

### MONK
    arch = [17, 10, 1]
    neuronsType = [InputNeuron, TanHNeuron, TanHNeuron]
    network = Network(arch, neuronsType)
    losses, misses = network.trainBFGS(training_patterns, training_labels, training_patterns, training_labels, lossObject, 100)
    scores = network.predict(training_patterns)

    plt.subplot(1,2,1)
    plt.plot(range(len(losses)), losses)
    plt.xlabel("epoch")
    plt.ylabel("Mean Squared Error")
    plt.subplot(1,2,2)
    plt.plot(range(len(misses)), misses)
    plt.xlabel("epoch")
    plt.ylabel("misclassification error")
    plt.tight_layout()
    plt.show()
### END MONK

### EXAMPLE
    ##' esempio
    arch = [2, 2, 2]
    neuronsType = [InputNeuron, SigmoidNeuron, OutputNeuron]
    network = Network(arch, neuronsType)
    #network.layers[1].neurons[0].weights = np.asarray([0.15, 0.2, 0.35])
    #network.layers[1].neurons[1].weights = np.asarray([0.25, 0.3, 0.35])
    #network.layers[2].neurons[0].weights = np.asarray([0.4, 0.45, 0.6])
    #network.layers[2].neurons[1].weights = np.asarray([0.5, 0.55, 0.6])

    #data = [[0.05, 0.1]]
    #target = [[0.01, 0.99]]
    data = [[i, i] for i in np.linspace(0, 1, 200)]
    target = [[np.sin(d[0]), np.square(d[1])] for d in data]

    #network.trainBFGS(data,target,data,target,lossObject, 50)

    predictions = network.predict(data)
    print "target example:", target
    print "predictions example:", predictions

    plt.plot([d[0] for d in data], [p[0] for p in predictions], label='f_prediction')
    plt.plot([d[1] for d in data], [p[1] for p in predictions], label='g_prediction')
    plt.plot([d[0] for d in data], [t[0] for t in target], label='f')
    plt.plot([d[1] for d in data], [t[1] for t in target], label='g')
    plt.legend(loc='best')
    #plt.show()

### END EXAMPLE

if __name__ == '__main__':
    main()