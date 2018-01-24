import pandas as pd
from Neural_network import *
import matplotlib.pyplot as plt
from Validation import *


def main():
    # read dataset
    df = pd.read_csv("../MLCup/ML-CUP17-TR.csv", comment='#', header=None)
    features_col = ["id","input1","input2","input3","input4","input5","input6","input7", "input8","input9","input10"]
    targets_col = ["target_x", "target_y"]
    df.columns = features_col + targets_col

    # shuffle dataset and holdout
    frac_tr, frac_vl, frac_ts = 0.5, 0.25, 0.25
    df = df.reindex(np.random.permutation(df.index))
    training_data = df[]


if __name__ == "__main__":
    main()
