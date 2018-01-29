import pandas as pd
from ml_cup import *

def main():

    # read dile train set
    df = pd.read_csv("../MLCup/ML-CUP17-TR_shuffled.csv", comment='#')
    features_col = ["input1","input2","input3","input4","input5","input6","input7", "input8","input9","input10"]
    targets_col = ["target_x", "target_y"]

    # divide pattern and targets
    pattern,labels = divide_patterns_labels(df,features_col,targets_col)

    # normalization objects used to normalize features (only the features!)
    normalizer = MinMaxScaler(feature_range=(-1, 1))
    x_scaled = normalizer.fit_transform(pattern)
    y_scaled = normalizer.fit_transform(labels)

    # divide in development set and test set
    development_patterns,development_labels, test_patterns, test_targets = holdout_cup(x_scaled,y_scaled, 0.9)

    #train model
    lossObject = EuclideanError(normalizer)

    #specify parameters
    architecture = [17,10,2]
    neurons = [InputNeuron,TanHNeuron,TanHNeuron]
    network = Network(architecture,neurons)
    epochs = 2
    learning_rate = 0.2
    batch_size = 1
    momentum = 0.6
    regularization = 0.001

    # train and get result
    squared_error, misClass_error, squared_error_test, misClass_error_test = network.train(
        data=development_patterns, targets=development_labels, eval_data=test_patterns,eval_targets=test_targets,
        lossObject=lossObject, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
        momentum=momentum, regularization=regularization)

    network.dump_weights("final_train_weigths_ml_cup")
    # getting average
    squared_error /= float(len(development_patterns))
    squared_error_test /= float(len(test_patterns))
    misClass_error /= float(len(development_patterns))
    misClass_error_test /= float(len(test_patterns))
    print_result(misClass_error, misClass_error_test, squared_error, squared_error_test,
                 architecture, batch_size, learning_rate, momentum, regularization, 1,
                 "test set", lossObject, "../image/")
    print "accuracy", 1-misClass_error[-1]
    print "accuracy test", 1-misClass_error_test[-1]
    print "squared error", squared_error[-1]
    print "squared error test", squared_error_test[-1]


    #open blind test
    dataset = pd.read_csv("../MLCup/ML-CUP17-TS.csv", comment='#', header=None)
    features_col = ["input1","input2","input3","input4","input5","input6","input7", "input8","input9","input10"]
    dataset.columns = ["id"] + features_col
    blind_dataset = dataset[features_col].values

    # make prediction
    prediction = network.predict(blind_dataset)
    print "prediction", prediction[0]

    #open another file
    file = open('../my_prediction.csv', 'w+')
    file.write("#Davide Italo Serramazza, Carlo Alessi\n#Lupari bianchi\n#ML-CUP2017\n03/02/2017\n")

    # filll it
    for i in range(0,len(prediction)):
        string = str(i+1)

        for j in prediction[i]:
            string += "," + str(j)

        file.write(string+"\n")



if __name__ == '__main__':
    main()