from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from LinearRegressionEqnSolver import LinearRegressionEqnSolver
from NonLinearRegWithFeatSel import NonLinearRegWithFeatSel
import time
import numpy


class RegressorPool:
    # General parameters
    random_state = None
    n_jobs = None
    max_iter = None
    runtime = None
    lin_solver = None
    new_reg = None

    # parameter for Linear regression
    linreg = None

    # parameter for RANSAC regression
    min_samples_ransac = None
    ransacreg = None

    # parameter for Lasso
    lambda_l1 = None
    lassoreg = None

    # parameter for Ridge
    lambda_l2 = None
    ridgereg = None

    # parameter for Decision Tree regression
    min_samples_split = None
    dtreg = None

    # parameter for Linear regression solver
    linregsolver = None

    # parameter for New regression method
    newreg = None

    def __init__(self, random_state, n_jobs, max_iter, min_samples_ransac, lambda_l1, lamba_l2,
                 min_samples_split, lin_solver, new_reg):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        if lin_solver and new_reg:
            self.runtime = numpy.zeros(7)
        elif lin_solver or new_reg:
            self.runtime = numpy.zeros(6)
        else:
            self.runtime = numpy.zeros(5)
        self.lin_solver = lin_solver
        self.new_reg = new_reg

        # Linear regressor
        self.linreg = LinearRegression(fit_intercept=False, n_jobs=self.n_jobs)

        # RANSAC regressor
        self.min_samples_ransac = min_samples_ransac
        self.ransacreg = RANSACRegressor(min_samples=self.min_samples_ransac, max_trials=self.max_iter,
                                         random_state=self.random_state)

        # Lasso regressor
        self.lambda_l1 = lambda_l1
        self.lassoreg = Lasso(alpha=self.lambda_l1, fit_intercept=False, max_iter=self.max_iter,
                              random_state=self.random_state)

        # Ridge regressor
        self.lambda_l2 = lamba_l2
        self.ridgereg = Ridge(alpha=self.lambda_l2, fit_intercept=False, max_iter=self.max_iter,
                              random_state=self.random_state)

        # Decision Tree regressor
        self.min_samples_split = min_samples_split
        self.dtreg = DecisionTreeRegressor(min_samples_split=self.min_samples_split, random_state=self.random_state)

        # Linear regression solver
        self.linregsolver = LinearRegressionEqnSolver()

        # New regression method
        self.newreg = NonLinearRegWithFeatSel(lambda_l1=self.lambda_l1, max_iter=self.max_iter,
                                              random_state=self.max_iter, n_jobs=self.n_jobs);

    def lin_reg_fit(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Linear Model..")
        self.linreg.fit(X=X, y=Y)
        self.runtime[0] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[0], 2)) + " seconds")

    def ransac_reg_fit(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting RANSAC Linear Model..")
        self.ransacreg.fit(X=X, y=Y)
        self.runtime[1] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[1], 2)) + " seconds")

    def lasso_reg_fit(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Linear Model with Lasso..")
        self.lassoreg.fit(X=X, y=Y)
        self.runtime[2] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[2], 2)) + " seconds")

    def ridge_reg_fit(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Linear Model with Ridge..")
        self.ridgereg.fit(X=X, y=Y)
        self.runtime[3] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[3], 2)) + " seconds")

    def dt_reg_fit(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Non-Linear Decision Tree Regressor Model..")
        self.dtreg.fit(X=X, y=Y)
        self.runtime[4] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[4], 2)) + " seconds")

    def lin_reg_solver(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Linear Model through solver..")
        self.linregsolver.fit(X=X, y=Y)
        self.runtime[5] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[5], 2)) + " seconds")

    def new_reg_fit(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting New regression model..")
        self.newreg.fit(X=X, y=Y)
        indx = 6
        if not self.lin_solver:
            indx = 5
        self.runtime[indx] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[indx], 2)) + " seconds")

    def fit_all(self, X, Y):
        self.lin_reg_fit(X, Y)
        self.ransac_reg_fit(X, Y)
        self.lasso_reg_fit(X, Y)
        self.ridge_reg_fit(X, Y)
        self.dt_reg_fit(X, Y)
        if self.lin_solver:
            self.lin_reg_solver(X, Y)
        if self.new_reg:
            self.new_reg_fit(X, Y)

    def predict_all(self, X):

        reg_predict_dict = {
            "lin_reg": self.linreg.predict(X),
            "ransac_reg": self.ransacreg.predict(X),
            "lasso_reg": self.lassoreg.predict(X),
            "ridge_reg": self.ridgereg.predict(X),
            "dt_reg": self.dtreg.predict(X),
            "lin_reg_solver": self.linregsolver.predict(X) if self.lin_solver else numpy.zeros(X.shape[0]),
            "neo_reg": self.newreg.predict(X) if self.new_reg else numpy.zeros(X.shape[0])
        }

        return reg_predict_dict
