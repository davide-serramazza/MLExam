import pandas as pd
from Neural_network import *
import matplotlib.pyplot as plt


def decode(data,encode):
    ris = []
    for i in range (len(data)):
        for j in range(1,encode[i]+1):
            if j==data[i]:
                ris.append(1)
            else:
                ris.append(0)
    return ris

def main():
    # 1. load dataset
    columns = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'id']
    train_data = pd.read_csv("../monk_datasets/monks-1.train", delim_whitespace=True, header=None)
    train_data.columns = columns
    print train_data.head()

    # 2. train neural network. set low learning rate because actual implementation is online
    network = Network(architecture=[17, 2, 1], neurons=[InputNeuron, SigmoidNeuron, SigmoidNeuron])
    tmp = train_data[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']].values

    #3. trasform encoding
    encoding = [3,3,2,3,4,2]
    patterns = []
    for i in range(len(tmp)):
        patterns.append(decode(tmp[i],encoding))
    labels = train_data["label"].values
    losses = network.train(data=patterns, targets=labels, epochs=1000, learning_rate=0.1,l=SquaredError(),
                           batch_size=1,momentum=0.1)

    # 4. visualize how loss changes over time
    #    plots changes a lot for different runs
    error = [i/len(patterns)*2 for i in losses]
    plt.plot(range(len(error)), error)
    plt.xlabel("epochs")
    plt.ylabel("misClassification")
    print min(error)
    plt.show()


if __name__ == "__main__":
    main()
