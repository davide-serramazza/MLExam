import pandas as pd
from Neural_network import *
import matplotlib.pyplot as plt


def main():
    # 1. load dataset
    columns = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'id']
    train_data = pd.read_csv("../monk_datasets/monks-1.train", delim_whitespace=True, header=None)
    train_data.columns = columns
    print train_data.head()

    # 3. train neural network. set low learning rate because actual implementation is online
    network = Network(architecture=[6, 2, 1], neurons=[InputNeuron, SigmoidNeuron, SigmoidNeuron])
    patterns = train_data[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']].values
    labels = train_data["label"].values
    losses = network.train(data=patterns, targets=labels, epochs=10000, learning_rate=0.01,l=MisClassified(),
                           batch_size=1,momentum=0.0)

    # 4. visualize how loss changes over time
    #    plots changes a lot for different runs
    error = [i/len(patterns)*2 for i in losses]
    plt.plot(range(len(error)), error)
    plt.xlabel("epochs")
    plt.ylabel("misClassification")
    plt.show()


if __name__ == "__main__":
    main()
