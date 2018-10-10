from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy
import time


class ClassifierPool:

    # General parameters
    random_state = None
    max_iter = None
    n_jobs = None
    runtime = None

    # Classifier specific parameters (Other parameters will be set to 'default' or a suitable value)
    # Decision tree parameters
    dt_min_split = None
    dt_classifier = None

    # Linear Support Vector Machine parameters
    lsvm_penalty = None
    lsvm_c = None
    lsvm_classifier = None

    # Non linear Support Vector Machine parameters
    nlsvm_c = None
    nlsvm_gma = None
    nlsvm_classifier = None

    # Perceptron parameters
    ptron_penalty = None
    ptron_c = None
    ptron_eta = None
    ptron_classifier = None

    # Logistic regression parameters
    logres_penalty = None
    logres_c = None
    logres_classifier = None

    # KNN classifier parameters
    knn_k = None
    knn_algo = None
    knn_classifier = None

    def __init__(self, n_jobs=4, max_iter=50, random_state=1, dt_min_split=10, lsvm_penalty='l2', lsvm_c=0.05,
                 nlsvm_c = 0.05,nlsvm_gma='auto', ptron_penalty='l2', ptron_c=0.05, ptron_eta=0.001, logres_penalty='l2',
                 logres_c=0.05, knn_k=3, knn_algo='kd_tree'):
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.random_state = random_state
        self.runtime = numpy.zeros(6)

        # Decision tree
        self.dt_min_split = dt_min_split
        # min_samples_leaf = 5 (because 5 samples is the min requirement for many sig test, e.g., chi2 test)
        self.dt_classifier = DecisionTreeClassifier(min_samples_split=self.dt_min_split, min_samples_leaf=5,
                                                    random_state=self.random_state)

        # Linear SVM
        self.lsvm_penalty = lsvm_penalty
        self.lsvm_c = lsvm_c
        # fit_intercept = False (scaled data will be provided)
        self.lsvm_classifier = LinearSVC(penalty=self.lsvm_penalty, C=self.lsvm_c, fit_intercept=False,
                                         random_state=self.random_state, max_iter=self.max_iter)

        # Non Linear SVM
        self.nlsvm_c = nlsvm_c
        self.nlsvm_gma = nlsvm_gma
        self.nlsvm_classifier = SVC(C=self.nlsvm_c, gamma=self.nlsvm_gma, max_iter=self.max_iter,
                                    random_state=self.random_state)

        # Perceptron
        self.ptron_penalty = ptron_penalty
        self.ptron_c = ptron_c
        self.ptron_eta = ptron_eta
        # fit_intercept = False (scaled data will be provided)
        self.ptron_classifier = Perceptron(penalty=self.ptron_penalty, alpha=self.ptron_c, fit_intercept=False,
                                           max_iter=self.max_iter, eta0=self.ptron_eta, random_state=self.random_state,
                                           n_jobs=self.n_jobs)

        # Logistic Regression
        self.logres_penalty = logres_penalty
        self.logres_c = logres_c
        # fit_intercept = False (scaled data will be provided)
        # solver = 'sag' (Fast stochastic averaged gradient descent)
        # multi_class = 'ovr' (Multi-class support)
        self.logres_classifier = LogisticRegression(penalty=self.logres_penalty, C=self.logres_c, fit_intercept=False,
                                                    random_state=self.random_state, solver='sag', n_jobs=self.n_jobs,
                                                    multi_class='multinomial', max_iter=self.max_iter)

        # KNN Classifier
        self.knn_k = knn_k
        self.knn_algo = knn_algo
        self.knn_classifier = KNeighborsClassifier(n_neighbors=self.knn_k, algorithm=self.knn_algo, p=2,
                                                   n_jobs=self.n_jobs)

    # fit Decision Tree
    def fit_dt(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Decision Tree Model..")
        self.dt_classifier.fit(X=X, y=Y)
        self.runtime[0] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[0], 2)) + " seconds")

    # fit Linear SVM
    def fit_lsvm(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Linear Support Vector Machine Model..")
        self.lsvm_classifier.fit(X=X, y=Y)
        self.runtime[1] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[1], 2)) + " seconds")

    # fit Non Linear SVM
    def fit_nlsvm(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Non-linear Support Vector Machine Model..")
        self.nlsvm_classifier.fit(X=X, y=Y)
        self.runtime[2] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[2], 2)) + " seconds")

    # fit Perceptron
    def fit_ptron(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Perceptron Model..")
        self.ptron_classifier.fit(X=X, y=Y)
        self.runtime[3] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[3], 2)) + " seconds")

    # fit Logistic Regression
    def fit_logress(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting Logistic Regression Model..")
        self.logres_classifier.fit(X=X, y=Y)
        self.runtime[4] = time.time() - start_time
        print("Fitting ended in " + "{0:.2f}".format(round(self.runtime[4], 2)) + " seconds")

    # fit KNN
    def fit_knn(self, X, Y):
        # Record fitting runtime
        start_time = time.time()
        print("Fitting KNN classification Model..")
        self.knn_classifier.fit(X=X, y=Y)
        self.runtime[5] = time.time() - start_time
        print("Fitting ended in "+"{0:.2f}".format(round(self.runtime[5], 2))+" seconds")

    # fit all 6 classifiers to the given data
    def fit_all_classifier(self, X, Y):
        self.fit_dt(X, Y)
        self.fit_lsvm(X, Y)
        self.fit_nlsvm(X, Y)
        self.fit_ptron(X, Y)
        self.fit_logress(X, Y)
        self.fit_knn(X, Y)

    # predict from all 6 classifiers
    def predict_all_classifier(self, X):

        classifier_predict_dict = {
            "dt": self.dt_classifier.predict(X=X),
            "lsvm": self.lsvm_classifier.predict(X=X),
            "nlsvm": self.nlsvm_classifier.predict(X=X),
            "ptron": self.ptron_classifier.predict(X=X),
            "logres": self.logres_classifier.predict(X=X),
            "knn": self.knn_classifier.predict(X=X)
        }
        return classifier_predict_dict

