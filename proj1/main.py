from SingleLayerNeuralNetwork import SingleLayerNeuralNetwork
import argparse
import pandas
import numpy
import copy
import matplotlib.pyplot as pyplot


def main():
    # setup args list
    arguments = setup_arguments()
    args = arguments.parse_args()

    #  'fit_option' and 'data_path' are mandatory parameters.
    if args.fit_option is None:
        print("No fitting algorithm specified, please check python3 main.py --help for options. Exiting..")
        raise SystemExit
    if args.data_path is None:
        print("No data path specified, please check python3 main.py --help for options. Exiting..")
        raise SystemExit
    if args.ds_name is None:
        print("No name specified for this data-set, please check python3 main.py --help for options. Exiting..")
        raise SystemExit

    # manual conversion for verbose as bool behaves in a weird way
    if args.verbose == 'FALSE':
        args.verbose = False
    else:
        args.verbose = True

    # random seed
    if args.seed is None:
        seed = 123
    else:
        seed = int(args.seed)

    # standardize the data-set
    if args.stdize is None:
        args.stdize = False
    elif args.stdize == 'TRUE':
        args.stdize = True
    else:
        args.stdize = False

    # create a slnn object
    slnnobject = SingleLayerNeuralNetwork(max_iter=1000 if args.max_iter is None else int(args.max_iter),
                                          eta=0.0001 if args.eta is None else float(args.eta),
                                          random_state=seed,
                                          fit_option=args.fit_option,
                                          min_cost=0 if args.min_cost is None else int(args.min_cost),
                                          verbose=False if args.verbose is None else args.verbose)

    X_train, X_test, Y_train, Y_test, classes = prepare_data(args.data_path, seed, args.stdize)

    # if number of unique class labels are greater than 2, then multi_fits and multi_predict would be used
    if len(classes) > 2:
        # deepcopy makes copies of object, rather than binding them to the same object
        # (which is the case with '=')
        slnnobj_array = [slnnobject]
        for i in range(1, len(classes)):
            slnnobj_array.append(copy.deepcopy(slnnobject))

        model = multi_fit(slnnobj_array, X_train, Y_train, classes)
    else:
        Y_mod = numpy.where(Y_train == classes[0], 1, -1)
        model = slnnobject.fit(X_train, Y_mod)

    # plot the cost reduction by epoch
    cost_rate(model, classes, args.fit_option, args.ds_name)

    # predict
    print("Testing the model on test data..")
    if len(classes) > 2:
        predictions = multi_predict(model, X_test, classes)
    else:
        predictions = slnnobject.predict(X_test)
        predictions = numpy.where(predictions == 1, classes[0], classes[1])

    # get accuracy
    accuracy = numpy.where(Y_test == predictions, 1, 0)
    accuracy = sum(accuracy) / len(Y_test)

    print(args.fit_option + " reported an accuracy of " + '{0:,.2f}'.format(accuracy) + " on "
          + args.ds_name + " data-set")


# prepare data
def prepare_data(data_path, seed, stdize):
    # get attributes and class labels for data
    _data = pandas.read_csv(data_path)
    _data = _data.dropna()  # Drop NaN rows

    X = _data.iloc[:, 0:(_data.shape[1] - 1)]

    # check for non-integer / non-float values
    X_dtypes = X.dtypes
    X_dtypes = (X_dtypes == int) | (X_dtypes == float)
    for i in range(0, X.shape[1]):
        if not X_dtypes[i]:
            old_entries = numpy.unique(X.iloc[:, i])
            X.iloc[:, i] = X.iloc[:, i].replace(to_replace=old_entries,
                                                value=list(range(0, len(old_entries))))
    # standardize X
    if stdize:
        print("Standardizing X by mean and sd..")
        X = standardize_data(X)

    Y = _data.iloc[:, _data.shape[1] - 1]
    Y = numpy.asarray(list(map(str, Y)))  # convert class labels to string
    classes = numpy.unique(Y)

    # split data into training and testing
    print("Splitting data, stratified on Y..")
    X_train, X_test, Y_train, Y_test = stratified_split_train_test(X, Y, test_size=0.3, classes=classes, seed=seed)

    return X_train, X_test, Y_train, Y_test, classes


# setup argument names and help options
def setup_arguments():
    arguments = argparse.ArgumentParser()

    arguments.add_argument('-max_iter', help='Maximum number of iterations')
    arguments.add_argument('-eta', help='Learning rate')
    arguments.add_argument('-ds_name', help='Name of the data-set [*]')
    arguments.add_argument('-fit_option', help='Option for learning model : perceptron, adaline or sgd [*]')
    arguments.add_argument('-min_cost', help='Minimum SSE error tolerance')
    arguments.add_argument('-data_path', help='Path to a csv file; Note: The last column will be treated '
                                              'as class labels and the rest as attributes [*]')
    arguments.add_argument('-verbose', help='Displays more information : FALSE, TRUE')
    arguments.add_argument('-seed', help='Random seed')
    arguments.add_argument('-stdize', help='Standardize the data? : FALSE, TRUE')

    return arguments


# split training and testing data
def stratified_split_train_test(X, Y, test_size, classes, seed):
    train_size = 1 - test_size
    rnd_div = numpy.random.RandomState(seed)

    train_index = []  # container for training samples
    test_index = []  # container for testing samples

    for i in range(0, len(classes)):
        all_class_ind = numpy.where(Y == classes[i])
        all_class_ind = sum(all_class_ind)  # convert tuples to list
        all_class_ind = all_class_ind[rnd_div.permutation(all_class_ind.size)]  # random permutation
        loc = int(all_class_ind.size*train_size)
        train_index.extend(all_class_ind[0:loc])  # extend training list
        test_index.extend(all_class_ind[loc:all_class_ind.size])  # extend testing list

    # subset the data-sets
    X_train = X.iloc[train_index, :]
    X_test = X.iloc[test_index, :]
    Y_train = Y[train_index]
    Y_test = Y[test_index]

    return X_train, X_test, Y_train, Y_test


# standardize data
def standardize_data(X):

    mu = X.mean(axis=0)  # compute mean per column
    sd = numpy.std(X, axis=0)  # compute standard deviation per column

    # Transform each column by mean and sd
    for i in range(0, X.shape[1]):
        X.iloc[:, i] = (X.iloc[:, i] - mu[i])/sd[i]

    return X


# fit a model for each class label
def multi_fit(slnnobj_array, X, Y, classes):
    slnn_models = []  # model collector
    iter = 0  # object iterator
    for _class in classes:
        print("Training model for class " + _class)
        Y_mod = numpy.where(Y == _class, 1, -1)  # one versus all
        model = slnnobj_array[iter].fit(X, Y_mod)
        slnn_models.append(model)
        iter = iter + 1
    return slnn_models


# predict using the fitted models
def multi_predict(model, X, classes):
    preds = numpy.zeros((X.shape[0], len(classes)))  # create an empty matrix
    for i in range(0, len(classes)):
        preds[:, i] = model[i].compute_net_input(X)
    pred = classes[numpy.argmax(preds, axis=1)]  # assign class label with highest regression/prediction value
    return pred


# plotting function
def cost_rate(model, classes, fit_option, ds_name):
    print("Plotting cost reduction charts..")
    pyplot.rcParams.update({'font.size': 22})  # changing font param
    if len(classes) > 2:  # multiple figures
        for itr in range(0, len(classes)):  # individual class labels
            pyplot.figure(num=itr + 1, figsize=(13, 9), dpi=200, facecolor='w', edgecolor='k')
            pyplot.plot(range(0, model[itr].max_iter), model[itr].cost_p_epoch, "-o")
            pyplot.title(fit_option + "\n" + classes[itr], fontsize=27)
            pyplot.xlabel("Epoch")
            if fit_option == 'perceptron':
                pyplot.ylabel("Mis-classifications")
            else:
                pyplot.ylabel("SSE")
            pyplot.savefig(ds_name + "_" + classes[itr] + "_" + fit_option + '_cost_redux.png')
    else:  # one figure
        pyplot.figure(num=1, figsize=(13, 9), dpi=200, facecolor='w', edgecolor='k')
        pyplot.plot(range(0, model.max_iter), model.cost_p_epoch, "-o")
        pyplot.title(fit_option, fontsize=27)
        pyplot.xlabel("Epoch")
        if fit_option == 'perceptron':
            pyplot.ylabel("Mis-classifications")
        else:
            pyplot.ylabel("SSE")
        pyplot.savefig(ds_name + "_" + fit_option + '_cost_redux.png')


main()
