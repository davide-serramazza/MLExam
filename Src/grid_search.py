from Neural_network import *
from utils import *

#### GRID SEARCH PARAMETERS ####

class GridSearchSGDParams:
    """
    specifies the hyperparameters to tune in SGD grid search
    """
    def __init__(self, learning_rate, momentum, batch_size, architecture, neurons, regularization, epoch):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.architecture = architecture
        self.neurons = neurons
        self.regularization = regularization
        self.epoch = epoch


class GridSearchBFGSParams:
    """
    specifies the hyperparameters to tune in BFGS grid search
    """
    def __init__(self, c_1, c_2, theta, regularization, epoch, architecture, neurons):
        self.c_1 = c_1
        self.c_2 = c_2
        self.theta = theta
        self.epoch = epoch
        self.architecture = architecture
        self.neurons = neurons
        self.regularization = regularization


class GridSearchLBFGSParams(GridSearchBFGSParams):
    """
    specifies the hyperparameters to tune in LBFGS grid search
    """
    def __init__(self, c_1, c_2, theta, regularization, m, epoch, architecture, neurons):
        GridSearchBFGSParams.__init__(self, c_1, c_2, theta, regularization, epoch, architecture, neurons)
        # number of vector used to define central matrix H
        self.m = m



#### GRID SEARCH IMPLEMENTATIONS ####

def grid_search_LBFGS(parameter, loss_obj, tr_patterns, tr_labels, vl_patterns, vl_labels, n_trials, save_in_dir):
    """
    grid search for optimal hyperparameter
    :param network: network to be trained
    :param loss_obj: loss used
    :param reguaritazion:
    :param n_trials: n of random trails for each value
    :param tr_patterns: traning set patterns
    :param tr_labels: traning set target
    :param vl_patterns: validation set patterns
    :param vl_labels: validation set target
    :param directory where to save results (learning curves)
    :return:
    """
    if not isinstance(parameter, GridSearchLBFGSParams):
        raise Exception("grid search parameters of class %s instead of GridSearchLBFGSParams", type(parameter))

    total_experiments = len(parameter.c_1) * len(parameter.c_2) \
                        * len(parameter.theta) * len(parameter.regularization) \
                        * len(parameter.m) * len(parameter.architecture)
    print "BEGIN GRID SEARCH L-BFGS: %d experiments" % total_experiments

    n_figure = 0  # index of figures
    # for every value
    for c_1 in parameter.c_1:
        for c_2 in parameter.c_2:
            for theta in parameter.theta:
                for reg in parameter.regularization:
                    for m in parameter.m:
                        for arc, neur in zip(parameter.architecture, parameter.neurons):
                            print n_figure, "out of", total_experiments, "experiments"
                            # initialize lists for saving reslut
                            squared_error_average = np.zeros(parameter.epoch + 1)
                            misClass_error_average = np.zeros(parameter.epoch + 1)
                            squared_error_validation_average = np.zeros(parameter.epoch + 1)
                            misClass_error_validation_average = np.zeros(parameter.epoch + 1)
                            # n_trials then average
                            for n in range(n_trials):
                                # buid a new network
                                network = Network(arc, neur)
                                # train
                                squared_error, misClass_error, \
                                squared_error_validation, misClass_error_validation = \
                                    network.trainLBFGS(data=tr_patterns, targets=tr_labels, eval_data=vl_patterns,
                                                       eval_targets=vl_labels,
                                                       lossObject=loss_obj, theta=theta, c_1=c_1, c_2=c_2, alpha_0=1,
                                                       m=m, epochs=parameter.epoch, regularization=reg)

                                # eventually pad vector
                                diff = squared_error_average.shape[0] - squared_error.shape[0]
                                squared_error = np.pad(squared_error, (0, diff), 'constant',
                                                       constant_values=(squared_error[-1]))
                                misClass_error = np.pad(misClass_error, (0, diff), 'constant',
                                                        constant_values=(misClass_error[-1]))

                                diff = squared_error_validation_average.shape[0] - squared_error_validation.shape[0]
                                squared_error_validation = np.pad(squared_error_validation, (0, diff), 'constant',
                                                                  constant_values=(squared_error_validation[-1]))
                                misClass_error_validation = np.pad(misClass_error_validation, (0, diff), 'constant',
                                                                   constant_values=(misClass_error_validation[-1]))

                                # append result of single epoch in list previously created
                                squared_error_average += squared_error
                                misClass_error_average += misClass_error
                                squared_error_validation_average += squared_error_validation
                                misClass_error_validation_average += misClass_error_validation


                                # taking mean error over trials and over patterns
                            squared_error_average /= float(n_trials)
                            misClass_error_average /= float(n_trials)
                            squared_error_validation_average /= float(n_trials) * len(vl_patterns)
                            misClass_error_validation_average /= float(n_trials) * len(vl_patterns)

                            print_result_LBFGS(misClass_error_average, misClass_error_validation_average,
                                               squared_error_average, squared_error_validation_average,
                                               arc, c_1, c_2, theta,
                                               reg, m, n_figure, "validation set", loss_obj, save_in_dir)
                            n_figure += 1  # increment to create a new figure


def grid_search_BFGS(parameter, loss_obj, tr_patterns, tr_labels, vl_patterns, vl_labels, n_trials, save_in_dir):
    """
    grid search for optimal hyperparameter
    :param network: network to be trained
    :param loss_obj: loss used
    :param reguaritazion:
    :param n_trials: n of random trails for each value
    :param tr_patterns: traning set patterns
    :param tr_labels: traning set target
    :param vl_patterns: validation set patterns
    :param vl_labels: validation set target
    :param directory where to save results (learning curves)
    :return:
    """
    if not isinstance(parameter, GridSearchBFGSParams):
        raise Exception("grid search parameters of class %s instead of GridSearchBFGSParams", type(parameter))

    total_experiments = len(parameter.c_1) * len(parameter.c_2) \
                        * len(parameter.theta) * len(parameter.regularization)
    print "BEGIN GRID SEARCH BFGS: %d experiments" % total_experiments
    n_figure = 0  # index of figures
    # for every value
    for c_1 in parameter.c_1:
        for c_2 in parameter.c_2:
            for theta in parameter.theta:
                for reg in parameter.regularization:
                    for arc, neur in zip(parameter.architecture, parameter.neurons):
                        print n_figure, "out of", total_experiments, "experiments"
                        # initialize lists for saving reslut
                        squared_error_average = np.zeros(parameter.epoch + 1)
                        misClass_error_average = np.zeros(parameter.epoch + 1)
                        squared_error_validation_average = np.zeros(parameter.epoch + 1)
                        misClass_error_validation_average = np.zeros(parameter.epoch + 1)
                        # n_trials then average
                        for n in range(n_trials):
                            # buid a new network
                            network = Network(arc, neur)
                            # train
                            squared_error, misClass_error, \
                            squared_error_validation, misClass_error_validation = \
                                network.trainBFGS(data=tr_patterns, targets=tr_labels, eval_data=vl_patterns,
                                                  eval_targets=vl_labels, lossObject=loss_obj, theta=theta, c_1=c_1,
                                                  c_2=c_2, epochs=parameter.epoch, regularization=reg)

                            # eventually pad vector
                            diff = squared_error_average.shape[0] - squared_error.shape[0]
                            squared_error = np.pad(squared_error, (0, diff), 'constant',
                                                   constant_values=(squared_error[-1]))
                            misClass_error = np.pad(misClass_error, (0, diff), 'constant',
                                                    constant_values=(misClass_error[-1]))

                            diff = squared_error_validation_average.shape[0] - squared_error_validation.shape[0]
                            squared_error_validation = np.pad(squared_error_validation, (0, diff), 'constant',
                                                              constant_values=(squared_error_validation[-1]))
                            misClass_error_validation = np.pad(misClass_error_validation, (0, diff), 'constant',
                                                               constant_values=(misClass_error_validation[-1]))

                            # append result of single epoch in list previously created
                            squared_error_average += squared_error
                            misClass_error_average += misClass_error
                            squared_error_validation_average += squared_error_validation
                            misClass_error_validation_average += misClass_error_validation

                            # taking mean error over trials and over patterns
                            squared_error_average /= float(n_trials)
                            misClass_error_average /= float(n_trials)
                            squared_error_validation_average /= float(n_trials) * len(vl_patterns)
                            misClass_error_validation_average /= float(n_trials) * len(vl_patterns)

                            print_result_BFGS(misClass_error_average, misClass_error_validation_average,
                                              squared_error_average, squared_error_validation_average,
                                              arc, c_1, c_2, theta,
                                              reg, n_figure, "validation set", loss_obj, save_in_dir)
                            n_figure += 1  # increment to create a new figure


def grid_search_SGD(parameter, loss_obj, tr_patterns, tr_labels, vl_patterns, vl_labels, n_trials, save_in_dir):
    """
    grid search for optimal hyperparameter
    :param network: network to be trained
    :param loss_obj: loss used
    :param reguaritazion:
    :param n_trials: n of random trails for each value
    :param tr_patterns: traning set patterns
    :param tr_labels: traning set target
    :param vl_patterns: validation set patterns
    :param vl_labels: validation set target
    :param directory where to save results (learning curves)
    :return:
    """
    if not isinstance(parameter, GridSearchSGDParams):
        raise Exception("grid search parameters of class %s instead of GridSearchSGDParams", type(parameter))

    total_experiments = len(parameter.regularization) * len(parameter.learning_rate) \
                        * len(parameter.momentum) * len(parameter.batch_size) * len(parameter.architecture)
    print "BEGIN GRID SEARCH SGD: %d experiments" % total_experiments

    n_figure = 0  # index of figures
    # for every value
    for reg in parameter.regularization:
        for lr in parameter.learning_rate:
            for mo in parameter.momentum:
                for bat in parameter.batch_size:
                    for arc, neur in zip(parameter.architecture, parameter.neurons):
                        print n_figure, "out of", total_experiments, "experiments"
                        # initialize lists for saving reslut
                        squared_error_average = np.zeros(parameter.epoch)
                        misClass_error_average = np.zeros(parameter.epoch)
                        squared_error_validation_average = np.zeros(parameter.epoch)
                        misClass_error_validation_average = np.zeros(parameter.epoch)
                        # n_trials then average
                        for n in range(n_trials):
                            # buid a new network
                            network = Network(arc, neur)
                            # train
                            squared_error, misClass_error, \
                            squared_error_validation, misClass_error_validation = \
                                network.train(data=tr_patterns, targets=tr_labels, eval_data=vl_patterns,
                                              eval_targets=vl_labels, lossObject=loss_obj, epochs=parameter.epoch,
                                              learning_rate=lr, batch_size=bat, momentum=mo, regularization=reg)

                            # append result of single epoch in list previously created
                            squared_error_average += squared_error
                            misClass_error_average += misClass_error
                            squared_error_validation_average += squared_error_validation
                            misClass_error_validation_average += misClass_error_validation

                        # taking mean error over trials and over patterns
                        squared_error_average /= float(n_trials) * len(tr_patterns)
                        misClass_error_average /= float(n_trials) * len(tr_patterns)
                        squared_error_validation_average /= float(n_trials) * len(vl_patterns)
                        misClass_error_validation_average /= float(n_trials) * len(vl_patterns)

                        print_result_SGD(misClass_error_average, misClass_error_validation_average,
                                         squared_error_average, squared_error_validation_average,
                                         arc, bat, lr, mo, reg, n_figure, "validation set", loss_obj, save_in_dir)
                        n_figure += 1  # increment to create a new figure