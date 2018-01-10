import pandas as pd
from Neural_network import *
import matplotlib.pyplot as plt

# decode inout from 1-to-k enconding
def decode(data,encode):
    ris = []
    for i in range (len(data)):
        for j in range(1,encode[i]+1):
            if j==data[i]:
                ris.append(1)
            else:
                ris.append(0)
    return ris

#transform output from 0 to -1 for Tanh
def transform_output(l):
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

    # 2. train neural network. set low learning rate because actual implementation is online
    network = Network(architecture=[17, 5, 5, 1], neurons=[InputNeuron, TanHNeuron, TanHNeuron, TanHNeuron])
    tmp = train_data[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']].values

    #3. trasform encoding
    encoding = [3,3,2,3,4,2]
    patterns = []
    for i in range(len(tmp)):
        patterns.append(decode(tmp[i],encoding))
    tmps = train_data["label"].values
    labels = transform_output(tmps)
    losses,misClass = network.train(data=patterns, targets=labels, epochs=100, learning_rate=0.01,
                                    batch_size=len(patterns), momentum=0.0, regularization=0.01)
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
    squareE = [2*i for i in losses]
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
    labels = transform_output(labels.values)

    print network.predict(test_patterns[:5]), labels[:5]
    print network.predict(patterns[:5]), labels[:5]


if __name__ == "__main__":
    main()
