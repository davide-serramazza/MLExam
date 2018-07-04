from ml_cup import *

def main():

    # 1. read file train set
    df = pd.read_csv("../MLCup/ML-CUP17-TR_shuffled.csv", comment='#')
    features_col = ["input1","input2","input3","input4","input5","input6","input7", "input8","input9","input10"]
    targets_col = ["target_x", "target_y"]

    # 2. divide pattern and targets
    patterns,labels = divide_patterns_labels(df,features_col,targets_col)

    # 3. divide in development set and test set
    development_patterns, development_labels, test_patterns, test_targets = holdout_cup(patterns,labels, 0.8)

    lossObject = EuclideanError(normalizer=None)

    # 4. define architecture and hyperparameters
    architecture = [10, 10, 2]
    neurons = [InputNeuron,SigmoidNeuron, OutputNeuron]
    network = Network(architecture,neurons)
    epochs = 200
    learning_rate = 0.2
    batch_size = 256
    momentum = 0.1
    regularization = 0.01

    # 5. train and get result
    losses_train = []
    losses_eval = []
    for e in range(epochs):
        error_train_epoch, _, error_evaluation_epoch, _ = network.train(
            data=development_patterns, targets=development_labels, eval_data=test_patterns, eval_targets=test_targets,
            lossObject=lossObject, epochs=1, learning_rate=learning_rate, batch_size=batch_size,
            momentum=momentum, regularization=regularization)

        losses_train.append(error_train_epoch)
        losses_eval.append(error_evaluation_epoch)

        with open("../weights/weights_final_model_epoch_%d.csv" % (e+1), "w") as out_file:
            network.dump_weights(out_file)

    # 6. getting average
    losses_train = np.array(losses_train) / float(len(development_patterns))
    losses_eval = np.array(losses_eval) / float(len(test_patterns))

    print "loss train:"
    print losses_train
    print "loss validation:"
    print losses_eval
    argmin_eval = np.argmin(losses_eval)
    print "epoch with least error on validation:", argmin_eval + 1
    print "MEE at epoch %d, (TR,VL) = %f, %f" % (argmin_eval + 1, losses_train[argmin_eval], losses_eval[argmin_eval])

    # 7. plot
    plt.plot(range(1, len(losses_train) + 1), losses_train, '--')
    plt.plot(range(1, len(losses_eval) + 1), losses_eval, '-')
    plt.legend(['training set', "validation set"])
    plt.xlabel("epochs")
    plt.ylabel(lossObject.__class__.__name__)
    s = "../image/MLCup/FINAL_model-" + "lr_" + transf_value(learning_rate) + \
        "-mo_" + transf_value(momentum) + "-bat:" + transf_value(batch_size) + \
        "-reg_" + transf_value(regularization) + "-arc_" + tranf_arc(architecture)
    plt.tight_layout()  # minimize overlap of subplots
    plt.savefig(s)

if __name__ == '__main__':
    main()