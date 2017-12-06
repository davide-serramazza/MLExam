import pandas as pd
from Neural_network import *
import matplotlib.pyplot as plt


def main():
    # 1. load dataset
    columns = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'id']
    train_data = pd.read_csv("../monk_datasets/monks-1.train", delim_whitespace=True, header=None)
    train_data.columns = columns

    # 2. create target concept for monk-1:  a1==a2 or a5==1
    #    replace True with 1 and False with 0
    train_data["monk-1"] = \
        [a == b or c == 1 for a, b, c in zip(train_data['f1'], train_data['f2'], train_data['f5'])]
    train_data["monk-1"].replace({True: 1, False: 0}, inplace=True)
    print train_data.head()

    # 3. train neural network. set low learning rate because actual implementation is online
    network = Network(architecture=[6, 2, 1], neurons=[InputNeuron, SigmoidNeuron, SigmoidNeuron])
    patterns = train_data[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']].values
    labels = train_data["monk-1"].values
    losses = network.train(data=patterns, targets=labels, epochs=100, learning_rate=0.01,l=MisClassified())

    # 4. visualize how loss changes over time
    #    plots changes a lot for different runs
    plt.plot(range(len(losses)), losses)
    plt.xlabel("epochs")
    plt.ylabel("SSE")
    plt.show()

if __name__ == "__main__":
    main()
