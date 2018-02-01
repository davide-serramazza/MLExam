import pandas as pd
from ml_cup import *

def main():
# open blind test
    dataset = pd.read_csv("../MLCup/ML-CUP17-TS.csv", comment='#', header=None)
    features_col = ["input1","input2","input3","input4","input5","input6","input7", "input8","input9","input10"]
    dataset.columns = ["id"] + features_col
    blind_dataset = dataset[features_col].values

    architecture = [10, 10, 2]
    neurons = [InputNeuron, SigmoidNeuron, OutputNeuron]
    network = Network(architecture, neurons)

    with open("../weights_final_ml.csv") as input_file:
        network.load_weights(input_file)

    # make prediction
    prediction = network.predict(blind_dataset)
    print "prediction", prediction[0]

    # open another file
    file = open('../Lupari_bianchi_ML-CUP17-TS.csv', 'w+')
    file.write("#Davide Italo Serramazza, Carlo Alessi\n#Lupari bianchi\n#ML-CUP2017\n#01/02/2017\n")

    # fill it
    for i in range(0, len(prediction)):
        string = str(i+1)

        for j in prediction[i]:
            string += "," + str(j)

        file.write(string+"\n")


if __name__ == '__main__':
    main()
