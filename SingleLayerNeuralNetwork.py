# imports
import numpy
from progress.bar import Bar


class SingleLayerNeuralNetwork:
    # class variables
    max_iter = None  # maximum number of iterations
    eta = None  # learning rate
    random_state = None  # random seed
    weights = None  # weights for each attribute/independent variable
    min_cost = None  # minimum SSE error tolerance
    verbose = None  # display additional info.
    fit_option = None  # learning model option : 'perceptron', 'adaline' or 'sgd'
    rndm_gen = None  # random number generator
    cost_p_epoch = None  # an array to store cost/error per epoch

    # Init function
    def __init__(self, max_iter, eta, random_state, fit_option, min_cost=0, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print("Setting class variables..")
        self.max_iter = max_iter
        self.eta = eta
        self.random_state = random_state
        self.fit_option = fit_option
        self.min_cost = min_cost
        self.rndm_gen = numpy.random.RandomState(self.random_state)
        self.cost_p_epoch = numpy.zeros(max_iter)

    # Assign random 'weights'
    def random_weight_assignment(self, X):
        if self.verbose:
            print(".... Generating random weights with seed " + str(self.random_state))
        self.weights = self.rndm_gen.normal(loc=0.0, scale=0.1, size=1 + X.shape[1])

    # Shuffle data
    def shuffle(self, X, Y):
        shuffled_index = self.rndm_gen.permutation(len(Y))
        return X.iloc[shuffled_index], Y[shuffled_index]

    # Compute the net input : W^tX dot product
    def compute_net_input(self, X):
        wtx = numpy.dot(X, self.weights[1:]) + self.weights[0]
        return wtx

    # Predict Y' given X
    def predict(self, X):
        wtx = self.compute_net_input(X)
        return numpy.where(wtx >= 0, 1, -1)

    # The 'Perceptron' algorithm to update 'weights'
    def perceptron_fit(self, X, Y):
        if self.verbose:
            print("Begin training data .. ")
            print(".... training on " + str(X.shape[0]) + " instances of " + str(X.shape[1]) + " attributes.")
        pb = Bar("Epochs", max=self.max_iter)
        self.random_weight_assignment(X)
        for itr in range(0, self.max_iter):
            pb.next()
            for i in range(0, X.shape[0]):
                term = self.eta * (Y[i] - self.predict(X.iloc[i, :]))
                self.weights[0] += term
                self.weights[1:] += term * X.iloc[i, :]
            error = Y - self.predict(X)  # Compute error
            cost = numpy.where(error == 0, 0, 1)
            cost = sum(cost)
            self.cost_p_epoch[itr] = cost
            if cost <= self.min_cost:  # If the false positive rate <= 'fpr' then stop
                break
        if itr < (self.max_iter-1):
            self.cost_p_epoch[(itr + 1):self.max_iter] = cost
        pb.finish()
        if self.verbose:
            print("Training ended with mis-classification rate of " + str(cost))
        return self

    # The 'Adeline' algorithm to update 'weights'
    def adeline_fit(self, X, Y):
        if self.verbose:
            print("Begin training data .. ")
            print(".... training on " + str(X.shape[0]) + " instances of " + str(X.shape[1]) + " attributes.")
        pb = Bar("Epochs", max=self.max_iter)
        self.random_weight_assignment(X)
        for itr in range(0, self.max_iter):
            pb.next()
            wtx = self.compute_net_input(X)
            error = numpy.subtract(Y, wtx)  # Compute error
            cost = sum(error ** 2) / 2
            self.cost_p_epoch[itr] = cost
            if cost <= self.min_cost:  # If the SSE <= 'min_cost' then stop
                break
            self.weights[1:] += self.eta * numpy.dot(X.T, error)
            self.weights[0] += self.eta * sum(error)
        if itr < (self.max_iter-1):
            self.cost_p_epoch[(itr + 1):self.max_iter] = cost
        pb.finish()
        if self.verbose:
            print("Training ended with SSE of " + str(cost))
        return self

    # The 'Stochastic gradient descent' algorithm to update 'weights'
    def sgd_fit(self, X, Y):
        if self.verbose:
            print("Begin training data .. ")
            print(".... training on " + str(X.shape[0]) + " instances of " + str(X.shape[1]) + " attributes.")
        pb = Bar("Epochs", max=self.max_iter)
        self.random_weight_assignment(X)
        for itr in range(0, self.max_iter):
            pb.next()
            X, Y = self.shuffle(X, Y)
            cost_p_inst = []
            for i in range(0, X.shape[0]):
                output = self.compute_net_input(X.iloc[i, :])
                error = Y[i] - output
                self.weights[1:] += self.eta * numpy.dot(X.iloc[i, :].T, error)
                self.weights[0] += self.eta * error
                error = 0.5 * error ** 2
                cost_p_inst.append(error)
            cost = sum(cost_p_inst)
            self.cost_p_epoch[itr] = cost
            if cost <= self.min_cost:  # If the SSE <= 'min_cost' then stop
                break
        if itr < (self.max_iter-1):
            self.cost_p_epoch[(itr + 1):self.max_iter] = cost
        pb.finish()
        if self.verbose:
            print("Training ended with SSE of " + str(cost))
        return self

    # The generic 'fit' method
    def fit(self, X, Y):
        if self.fit_option == 'perceptron':
            if self.verbose:
                print("Fitting Perceptron..")
            return self.perceptron_fit(X, Y)
        elif self.fit_option == 'adeline':
            if self.verbose:
                print("Fitting Adeline..")
            return self.adeline_fit(X, Y)
        elif self.fit_option == 'sgd':
            if self.verbose:
                print("Fitting Stochastic Gradient Descent..")
            return self.sgd_fit(X, Y)
        else:
            if self.verbose:
                print("Incorrect option, please choose from 'perceptron', 'adeline' or 'sgd'")