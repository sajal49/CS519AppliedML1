import numpy


class LinearRegressionEqnSolver:
    weights = None

    def __init__(self):
        pass

    def fit(self, X, y):

        ones = numpy.ones((X.shape[0]))
        ones = ones.reshape(ones.shape[0], 1)
        self.weights = numpy.zeros(X.shape[1])

        X_mod = numpy.hstack((ones, X))
        self.weights = numpy.dot(numpy.linalg.inv(numpy.dot(X_mod.T, X_mod)),
                                 numpy.dot(X_mod.T, y))

    def predict(self, X):

        pred = self.weights[0] + numpy.dot(self.weights[1:], X.T)
        return pred
