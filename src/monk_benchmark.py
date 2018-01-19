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
    print "before shuffle\n", train_data.head()

    while True:
        # shuffle data set
        train_data = train_data.reindex(np.random.permutation(train_data.index))
        labels, patterns = get_patterns_and_labels(train_data)
        lossObject = SquaredError("tangentH")
        #4. hold out
        tr_patterns,tr_labels,vl_patterns,vl_labels = Validation.hold_out(patterns,labels,0.7)
        if np.absolute(np.sum(vl_labels)) <4:
            break

    # validation: define hyperparameters to test
    architecture = [ [17, 10, 1]]
    neurons= [[InputNeuron, TanHNeuron, TanHNeuron], [InputNeuron, TanHNeuron, TanHNeuron,TanHNeuron] ]
    momentum = [0.4, 0.5, 0.6]
    batch_size = [10]
    learning_rate = [0.15, 0.2, 0.25]
    param = Validation.grid_search_parameter(learning_rate,momentum,batch_size,architecture,neurons)
    Validation.grid_search(param,lossObject,tr_patterns,tr_labels,vl_patterns,vl_labels, n_trials=5)


def get_patterns_and_labels(train_data):
    tmp = train_data[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']].values
    # 3. trasform encoding
    encoding = [3, 3, 2, 3, 4, 2]
    patterns = []
    for i in range(len(tmp)):
        patterns.append(decode(tmp[i], encoding))
    tmps = train_data["label"].values
    # transform output
    labels = transform_target(tmps)
    return labels, patterns


"""
    losses, misClass,_,_= network.train(data=patterns, targets=labels, vl_data=[], vl_targets=[],
                                     lossObject=lossObject, epochs=300, learning_rate=0.01,
                                    batch_size=1, momentum=0.0, regularization=0.01)

    misClass = np.array(misClass) / float(len(patterns))


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
