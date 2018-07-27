import matplotlib.pyplot as plt
import numpy as np

#### transform values to string format (to create file name of image) #####

def transf_value(value):
    """
    transform value of variable to suitable string
    :param value: string to be transformef
    :return:
    """
    return str(value).replace(".",",")


def tranf_arc(architecture):
    """
    transoform architecture value to suitable string
    :param architecture: architecture to be transformed
    :return:
    """
    s = "["
    for i in architecture:
        s += str(i)+","
    s += "]"
    return s


###### to print results of GRID SEARCH ######

def print_result_LBFGS(misClass_error, misClass_error_evaluation,
                 squared_error, squared_error_evaluation, arc,c_1,c_2,theta, reg, m, n_figure,
                 eval_string, lossObject, save_in_dir):
    # get accuracy
    accuracy = 1 - misClass_error
    accuracy_avarage = 1 - misClass_error_evaluation
    # plot result
    plt.figure(n_figure, dpi=300)  # select figure number 'n_figure'
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(accuracy) + 1), accuracy, '--')
    plt.plot(range(1, len(accuracy_avarage) + 1), accuracy_avarage, '-')
    plt.legend(['training set', eval_string])
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    # plot squaredError
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(squared_error) + 1), squared_error, '--')
    plt.plot(range(1, len(squared_error_evaluation) + 1), squared_error_evaluation, '-')
    plt.legend(['training set', eval_string])
    plt.xlabel("epochs")
    plt.ylabel(lossObject.__class__.__name__)
    s = save_in_dir + "c1_" + transf_value(c_1) + "-c2_" + transf_value(
        c_2) + "-theta_" + transf_value(theta) + "-reg_" + transf_value(reg) + "-m_" + str(m) +\
        "-arc_" + tranf_arc(arc)
    plt.tight_layout()  # minimize overlap of subplots
    plt.savefig(s)
    plt.close()
    print s, "got MEE (TR/VL)", squared_error[-1], squared_error_evaluation[-1]


def print_result_BFGS(misClass_error, misClass_error_evaluation,
                 squared_error, squared_error_evaluation, arc,c_1,c_2,theta, reg, n_figure,
                 eval_string, lossObject, save_in_dir):
    # get accuracy
    accuracy = 1 - misClass_error
    accuracy_avarage = 1 - misClass_error_evaluation
    # plot result
    plt.figure(n_figure, dpi=300)  # select figure number 'n_figure'
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(accuracy) + 1), accuracy, '--')
    plt.plot(range(1, len(accuracy_avarage) + 1), accuracy_avarage, '-')
    plt.legend(['training set', eval_string])
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    # plot squaredError
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(squared_error) + 1), squared_error, '--')
    plt.plot(range(1, len(squared_error_evaluation) + 1), squared_error_evaluation, '-')
    plt.legend(['training set', eval_string])
    plt.xlabel("epochs")
    plt.ylabel(lossObject.__class__.__name__)
    s = save_in_dir + "c1_" + transf_value(c_1) + "-c2_" + transf_value(
        c_2) + "-theta_" + transf_value(theta) + "-reg_" + transf_value(reg) +\
        "-arc_" + tranf_arc(arc)
    plt.tight_layout()  # minimize overlap of subplots
    plt.savefig(s)
    plt.close()
    print s, "got MEE (TR/VL)", squared_error[-1], squared_error_evaluation[-1]


def print_result_SGD(misClass_error, misClass_error_evaluation,
                 squared_error, squared_error_evaluation, arc, bat, lr, mo,
                 reg, n_figure, eval_string, lossObject, save_in_dir):
    # get accuracy
    accuracy = 1 - misClass_error
    accuracy_avarage = 1 - misClass_error_evaluation
    # plot result
    plt.figure(n_figure, dpi=300)  # select figure number 'n_figure'
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(accuracy) + 1), accuracy, '--')
    plt.plot(range(1, len(accuracy_avarage) + 1), accuracy_avarage, '-')
    plt.legend(['training set', eval_string])
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    # plot squaredError
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(squared_error) + 1), squared_error, '--')
    plt.plot(range(1, len(squared_error_evaluation) + 1), squared_error_evaluation, '-')
    plt.legend(['training set', eval_string])
    plt.xlabel("epochs")
    plt.ylabel(lossObject.__class__.__name__)
    s = save_in_dir + "lr_" + transf_value(lr) + "-mo_" + transf_value(mo) + "-bat:" + transf_value(
        bat) + "-reg_" + transf_value(reg) + "-arc_" + tranf_arc(arc)
    plt.tight_layout()  # minimize overlap of subplots
    plt.savefig(s)
    plt.close()
    print s, "got MEE (TR/VL)", squared_error[-1], squared_error_evaluation[-1]


#### ONE-HOT ENCODING AND OTHER TRANSFORMATIONS USED IN MONK BENCHMARK  ######

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
        if i == 0:
            res.append(np.array([-1]))
        else:
            res.append(np.array([1]))
    return res


def transform_labels(training_set, validation_set):
    training_labels = transform_target(training_set["label"].values)
    validation_labels = transform_target(validation_set["label"].values)
    return training_labels, validation_labels


def decode_patterns(encoding, features, training_set, validation_set):
    training_patterns = [decode(pattern, encoding) for pattern in training_set[features].values]
    validation_patterns = [decode(pattern, encoding) for pattern in validation_set[features].values]
    return training_patterns, validation_patterns


#### PREPROCESSING ML CUP DATASET #####

def divide_patterns_labels(partition, feature_col, target_col):
    patterns = partition[feature_col].values
    labels = partition[target_col].values
    return patterns, labels


def holdout_cup(patterns, labels, frac_tr):
    # calculate size
    len_partion = int(frac_tr * len(patterns))

    # divide train/set
    first_partition_patterns = patterns[:len_partion]
    first_partition_labels = labels[:len_partion]
    second_partition_pattens = patterns[len_partion:]
    second_partition_labels = labels[len_partion:]
    return first_partition_patterns, first_partition_labels, second_partition_pattens, second_partition_labels