from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy, time


class EnsemblePool:

    # common param
    random_state = None
    n_jobs = None
    n_estimators = None
    min_samples_leaf = None
    runtime = None

    # Base DT param
    base_dt_obj = None

    # Adaboost param
    adab_obj = None

    # Random Forest param
    rf_obj = None

    # Bagging param
    bgg_obj = None

    def __init__(self, random_state, n_jobs, n_estimators, min_samples_leaf):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.runtime = numpy.zeros(4)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf

        # Base Decision Tree classifier
        self.base_dt_obj = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf,
                                                  random_state=self.random_state)
        # Adaboost classifier
        self.adab_obj = AdaBoostClassifier(base_estimator=self.base_dt_obj, n_estimators=self.n_estimators,
                                           random_state=self.random_state)
        # Random Forest Classifier
        self.rf_obj = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                                             n_jobs=self.n_jobs, random_state=self.random_state)

        # Baggin Classifier
        self.bgg_obj = BaggingClassifier(base_estimator=self.base_dt_obj, n_estimators=self.n_estimators,
                                         n_jobs=self.n_jobs, random_state=self.random_state)

    def DT_fit(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Base Decision Tree model")
        self.base_dt_obj.fit(X=X, y=Y)
        self.runtime[0] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[0], 2)) + " seconds")

    def ADAB_fit(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting ADABOOST Decision Tree model")
        self.adab_obj.fit(X=X, y=Y)
        self.runtime[1] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[1], 2)) + " seconds")

    def RF_fit(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Random Forest model")
        self.rf_obj.fit(X=X, y=Y)
        self.runtime[2] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[1], 2)) + " seconds")

    def BAGG_fit(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Bagging Decision Tree model")
        self.bgg_obj.fit(X=X, y=Y)
        self.runtime[3] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[1], 2)) + " seconds")

    def Fit_all(self, X, Y):
        self.DT_fit(X, Y)
        self.ADAB_fit(X, Y)
        self.RF_fit(X, Y)
        self.BAGG_fit(X, Y)

    def Predict_all(self, X):

        predictions_dict = {
            'Base DT': self.base_dt_obj.predict(X),
            'ADAB DT': self.adab_obj.predict(X),
            'RF': self.rf_obj.predict(X),
            'BAG DT': self.bgg_obj.predict(X)
        }

        return predictions_dict


