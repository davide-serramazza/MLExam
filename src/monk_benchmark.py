import pandas as pd
from Neural_network import *
import matplotlib.pyplot as plt
import Validation

def decode(data,encoding):
    """
    Decode examples encoded with 1-of-k

    :param data: vector of examples to decode
    :param encoding: 1-of-k encoding used
    :return: decoded data
    """
    ris = []
    for i in range (len(data)):
        for j in range(1,encoding[i]+1):
            if j==data[i]:
                ris.append(1)
            else:
                ris.append(0)
    return ris

def transform_target(l):
    """
    transform specific negative example's target from 0 to -1
    (needed if using tanH as output)

    :param l: vector containing targets
    :return: transformed vector targets
    """
    res = []
    for i in l:
        if i==0:
            res.append(-1)
        else:
            res.append(1)
    return res

def main():
    # 1. load dataset
    columns = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'id']
    train_data = pd.read_csv("../monk_datasets/monks-1.train", delim_whitespace=True, header=None)
    train_data.columns = columns
    print train_data.head()
    # shuffle data set
    train_data = train_data.reindex(np.random.permutation(train_data.index))
    print "after shffle", train_data.head()

    # 2. train neural network. set low learning rate because actual implementation is online
    network = Network(architecture=[17, 5, 5, 1], neurons=[InputNeuron, TanHNeuron, TanHNeuron, TanHNeuron])
    tmp = train_data[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']].values

    #3. trasform encoding
    encoding = [3,3,2,3,4,2]
    patterns = []
    for i in range(len(tmp)):
        patterns.append(decode(tmp[i], encoding))
    tmps = train_data["label"].values
    labels = transform_target(tmps)
    lossObject = SquaredError("tangentH")
    #4. hold out
    Validation.hold_out(network,lossObject,patterns,labels,0.7)
    """
    losses, misClass = network.train(data=patterns, targets=labels,lossObject=lossObject, epochs=100, learning_rate=0.01,
                                    batch_size=1, momentum=0.0, regularization=0.01)
    misClass = np.array(misClass) / len(patterns)
    # TODO problemi con la regolarizzazione


    # 4. visualize how loss changes over time
    #    plots changes a lot for different runs
    #todo specify graph/window dimension
    plt.subplot(1, 2, 1)
    plt.plot(range(len(misClass)), misClass)
    plt.xlabel("epochs")
    plt.ylabel("misClassification")
    #plt.show()
    #plot squaredError
    plt.subplot(1,2,2)
    squareE = np.asarray(losses) * 2 / len(losses)  # TODO: divide by len(losses) to obtain MSE?
    plt.plot(range(len(squareE)),squareE)
    plt.xlabel("epochs")
    plt.ylabel("squaredError")
    plt.show()

    # predict
    test_data = pd.read_csv("../monk_datasets/monks-1.test", delim_whitespace=True, header=None)
    test_data.columns = columns
    labels = test_data["label"]
    test_data = test_data[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']].values
    test_patterns = []
    for i in range(len(test_data)):
        test_patterns.append(decode(test_data[i], encoding))
    labels = transform_target(labels.values)

    scores = network.predict(test_patterns)
    print scores[:10], labels[:10]
"""

if __name__ == "__main__":
    main()
