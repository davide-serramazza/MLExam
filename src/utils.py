import matplotlib.pyplot as plt
import numpy as np

def plot_condition_number_vs_iterations(cond_bfgs=None, cond_lbfgs=None):
    if cond_bfgs is not None:
        plt.plot(range(len(cond_bfgs)), cond_bfgs, '--', alpha=0.8, label='bfgs', linewidth=2)
    if cond_lbfgs is not None:
        plt.plot(range(len(cond_lbfgs)), cond_lbfgs, '-', alpha=0.8, label='l-bfgs', linewidth=2)
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterazioni')
    plt.ylabel(r'$k(H)$')
    plt.show()

def lbfgs_iterations_vs_m(iter_values, m_values):
    plt.yscale('log')
    plt.plot(m_values, iter_values, '-o', alpha=1)
    plt.xlabel('m')
    plt.ylabel('iterazioni')
    plt.show()

def shuffle_dataset(data, targets):
    permutation = np.random.permutation(len(data))
    data_shuffled = [data[i] for i in permutation]
    targets_shuffled = [targets[i] for i in permutation]
    return data_shuffled, targets_shuffled


def plot_train_test_learning_curve_loss(loss_test, loss_train):
    """
    plots loss (in log-scale) and accuracy learning curves on training and test set
    :param loss_test:
    :param loss_train:
    :param misclass_test:
    :param misclass_train:
    :return:
    """
    plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(range(len(loss_train)), loss_train, '-o', alpha=0.8, label='train loss', linewidth=2, color='blue')
    plt.plot(range(len(loss_test)), loss_test, '-x', alpha=0.8, label='test loss', linewidth=2, color='green')
    plt.legend(loc='best')
    plt.xlabel('Iterazioni')
    plt.ylabel('Errore')
    plt.show()

def plot_train_test_learning_curve_accuracy(misclass_train, misclass_test):
    plt.figure()
    plt.plot(range(len(misclass_train)), 1 - misclass_train, '-o', alpha=0.8, label='train accuracy', linewidth=2, color='blue')
    plt.plot(range(len(misclass_test)), 1 - misclass_test, '-x', alpha=0.8, label='test accuracy', linewidth=2, color='green')
    plt.legend(loc='best')
    plt.xlabel('Iterazioni')
    plt.ylabel('Accuratezza')
    plt.show()

def plot_norm_gradient_vs_iterations(g_norm_list_sgd=None, g_norm_list_bfgs=None, g_norm_list_lbfgs=None):
    if g_norm_list_sgd is not None:
        plt.plot(range(len(g_norm_list_sgd)), g_norm_list_sgd, '-', alpha=0.8, label='sgd', color='red', linewidth=2)
    if g_norm_list_bfgs is not None:
        plt.plot(range(len(g_norm_list_bfgs)), g_norm_list_bfgs, '--', alpha=0.8, label='bfgs', color='black', linewidth=2)
    if g_norm_list_lbfgs is not None:
        plt.plot(range(len(g_norm_list_lbfgs)), g_norm_list_lbfgs, ':', alpha=0.8, label='l-bfgs', color='blue', linewidth=2)

    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterazioni')
    plt.ylabel(r'$||\nabla E(w)||$')
    plt.show()


def plot_alpha_vs_iterations(alpha_list_bfgs=None, alpha_list_lbfgs=None):
    if alpha_list_bfgs is not None:
        plt.step(range(len(alpha_list_bfgs)), alpha_list_bfgs, 'o', alpha=0.8, label='bfgs', color='black')
    if alpha_list_lbfgs is not None:
        plt.step(range(len(alpha_list_lbfgs)), alpha_list_lbfgs, 's', alpha=0.8, label='l-bfgs', color='blue')
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterazioni')
    plt.ylabel('step size ' + r'$\alpha$')
    plt.show()


def plot_relative_gap_vs_iterations(loss_tr_sgd=None, loss_tr_bfgs=None, loss_tr_lbfgs=None):
    labels = ['sgd', 'bfgs', 'l-bfgs']
    line_style = ['-', '--', ':']
    color = ['red', 'black', 'blue']
    lists = [loss_tr_sgd, loss_tr_bfgs, loss_tr_lbfgs]

    for l, style, label, c in zip(lists, line_style, labels, color):
        if l is not None:
            l_ = np.array(l)
            value_at_x_star = l_[-1]
            relative_gap_list = np.abs(l_ - value_at_x_star)
            plt.plot(range(len(relative_gap_list)), relative_gap_list, style, alpha=0.8, label=label, color=c, linewidth=2)

    plt.legend(loc='best')
    plt.xlabel('Iterazioni')
    plt.ylabel(r'$|E(w) - E(w*)|$')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()


def plot_all_loss(sgd_tr, sgd_ts, bfgs_tr, bfgs_ts, lbfgs_tr, lbfgs_ts, \
                    xscale='linear', yscale='log'):
    labels = ['sgd_tr', 'sgd_ts', 'bfgs_tr', 'bfgs_ts', 'lbfgs_tr', 'lbfgs_ts']
    line_style = ['-o', '--o', '-x', '--x', '-*', '--*']
    lists = [sgd_tr, sgd_ts, bfgs_tr, bfgs_ts, lbfgs_tr, lbfgs_ts]

    plt.figure()
    for l, style, label in zip(lists, line_style, labels):
        plt.plot(range(len(l)), l, style, alpha=0.8, label=label)
    plt.legend(loc='best')
    plt.xlabel('Iterazioni')
    plt.ylabel(r'$E(w)$')
    plt.yscale(yscale)
    plt.xscale(xscale)
    plt.show()

def plot_all_accuracy(sgd_tr, sgd_ts, bfgs_tr, bfgs_ts, lbfgs_tr, lbfgs_ts, \
                    xscale='linear', yscale='linear'):
    labels = ['sgd_tr', 'sgd_ts', 'bfgs_tr', 'bfgs_ts', 'lbfgs_tr', 'lbfgs_ts']
    line_style = ['-o', '--o', '-x', '--x', '-*', '--*']
    lists = [sgd_tr, sgd_ts, bfgs_tr, bfgs_ts, lbfgs_tr, lbfgs_ts]

    plt.figure()
    for l, style, label in zip(lists, line_style, labels):
        plt.plot(range(len(l)), 1 - l, style, alpha=0.8, label=label)
    plt.legend(loc='best')
    plt.xlabel('Iterazioni')
    plt.ylabel('Accuratezza')
    plt.yscale(yscale)
    plt.xscale(xscale)
    plt.show()

#### transform values to string format (to create file name of image) #####

def transf_value(value):
    """
    transform value of variable to suitable string
    :param value: string to be transformed
    :return:
    """
    return str(value).replace(".",",")


def tranf_arc(architecture):
    """
    transform architecture value to suitable string
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
                 squared_error, squared_error_evaluation, arc,c_1,c_2,theta, reg, epsilon, m, n_figure,
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
        c_2) + "-theta_" + transf_value(theta) + "-reg_" + transf_value(reg) + "-eps_" + \
        transf_value(epsilon) + "-m_" + str(m) + "-arc_" + tranf_arc(arc)
    plt.tight_layout()  # minimize overlap of subplots
    plt.savefig(s)
    plt.close()
    print s, "got MEE (TR/VL)", squared_error[-1], squared_error_evaluation[-1]


def print_result_BFGS(misClass_error, misClass_error_evaluation,
                 squared_error, squared_error_evaluation, arc,c_1,c_2,theta, reg, epsilon, n_figure,
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
        "-eps_" + transf_value(epsilon) + "-arc_" + tranf_arc(arc)
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

def decode(data, encoding):
    """
    Decode examples encoded with 1-of-k

    :param data: vector of examples to decode
    :param encoding: 1-of-k encoding used
    :return: decoded data
    """
    ris = []
    for i in range(len(data)):
        for j in range(1, encoding[i]+1):
            if j == data[i]:
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


def holdout_cup(patterns, labels, fraction_tr):
    # calculate size
    len_partion = int(fraction_tr * len(patterns))

    # divide train/set
    first_partition_patterns = patterns[:len_partion]
    first_partition_labels = labels[:len_partion]
    second_partition_pattens = patterns[len_partion:]
    second_partition_labels = labels[len_partion:]
    return first_partition_patterns, first_partition_labels, second_partition_pattens, second_partition_labels

def is_pos_def(x):
    """
    checks if matrix 'x' is positive semidefinite. For debugging and exploration reasons
    prints out if at least one eigenvalue is equal to 0.
    """
    eigenvalues = np.linalg.eigvals(x)
    if np.any(eigenvalues < 1e-16):
        print "eigenvalues - at least one equal to 0"

    return np.all(eigenvalues > 1e-16)

def check_dimensions(architecture, x_train, y_train):
    if architecture[0] != len(x_train[0]):
        raise Exception("network input dimension (%d) and input data dimension (%d) must match!" %(architecture[0], len(x_train[0])))
    if architecture[-1] != len(y_train[0]):
        raise Exception("network output dimension (%d) and target data dimension (%d) must match!" %(architecture[0], len(y_train[0])))
