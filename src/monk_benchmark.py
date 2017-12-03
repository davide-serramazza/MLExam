import pandas as pd
from Neural_network import *


def main():
    columns = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'id']
    train_data = pd.read_csv("../monk_datasets/monks-1.train", delim_whitespace=True, header=None)
    train_data.columns = columns
    print train_data.head()

    #network = Network(architecture=[])

if __name__ == "__main__":
    main()
