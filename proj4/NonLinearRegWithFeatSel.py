import numpy
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


class NonLinearRegWithFeatSel:

    weights = None
    max_iter = None
    random_state = None
    n_jobs = None
    lambda_l1 = None
    lassocv = None
    l2reg = None
    indx_f_only = None
    indx_int_only = None

    def __init__(self, lambda_l1, max_iter, random_state, n_jobs):
        self.lambda_l1 = lambda_l1
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lassocv = LassoCV(max_iter=self.max_iter, cv=10, n_jobs=self.n_jobs,
                               random_state=self.random_state)
        self.l2reg = Ridge(alpha=1, fit_intercept=False, max_iter=self.max_iter,
                           random_state=self.random_state)
        self.indx_f_only = []
        self.indx_int_only = []

    def fit(self, X, y):

        # features only
        new_features = numpy.ones(X.shape[0])
        new_features = new_features.reshape(X.shape[0], 1)

        for f in range(0, X.shape[1]):
            X1 = X[:,f]
            X1 = X1.reshape(X.shape[0], 1)
            X2 = X1 ** 2
            X_f = numpy.hstack((X1, X2))
            self.lassocv.fit(X_f, y)
            recruit = sum(numpy.where(numpy.around(self.lassocv.coef_, 2) != 0))
            self.indx_f_only.append(recruit)
            if len(recruit) != 0:
                new_features = numpy.hstack((new_features, X_f[:, recruit]))

        # interaction features
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        X_f = poly.fit_transform(X)
        int_indx = sum(numpy.where(numpy.sum(a=poly.powers_, axis=1)==2))
        X_f = X_f[:, int_indx]
        self.lassocv.fit(X_f, y)
        recruit = sum(numpy.where(numpy.around(self.lassocv.coef_, 2) != 0))
        self.indx_int_only = poly.powers_[int_indx][recruit]
        if len(recruit) != 0:
            new_features = numpy.hstack((new_features, X_f[:, recruit]))
        new_features = numpy.delete(new_features, 0, 1)
        if len(new_features) == 0:
            self.l2reg.fit(X=X, y=y)
        else:
            self.l2reg.fit(X=new_features, y=y)

    def predict(self, X):

        poly = PolynomialFeatures(degree=2, interaction_only=True)
        X_int = poly.fit_transform(X)

        new_features = numpy.ones(X.shape[0])
        new_features = new_features.reshape(X.shape[0], 1)

        # features
        f = 0
        for i in self.indx_f_only:
            if len(i) == 2:
                X1 = X[:, f]
                X1 = X1.reshape(X.shape[0], 1)
                X2 = X1 ** 2
                X_f = numpy.hstack((X1, X2))
                new_features = numpy.hstack((new_features, X_f))
            if len(i) == 1:
                if i[0] == 0:
                    X1 = X[:, f]
                    X1 = X1.reshape(X.shape[0], 1)
                    new_features = numpy.hstack((new_features, X1))
                elif i[0] == 1:
                    X1 = X[:, f]
                    X1 = X1.reshape(X.shape[0], 1)
                    X2 = X1 ** 2
                    new_features = numpy.hstack((new_features, X2))
            f = f+1

        # interactions
        for i in self.indx_int_only:
            indx = sum(numpy.where(i == 1))
            X_int = X[:,indx[0]] * X[:,indx[1]]
            X_int = X_int.reshape(X.shape[0], 1)
            new_features = numpy.hstack((new_features, X_int))

        new_features = numpy.delete(new_features, 0, 1)
        if len(new_features) == 0:
            pred = self.l2reg.predict(X)
        else:
            pred = self.l2reg.predict(X=new_features)

        print(pred)


