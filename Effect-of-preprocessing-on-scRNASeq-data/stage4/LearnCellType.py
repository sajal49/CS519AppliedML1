from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class LearnCellType:

    dt_learn = None
    knn_learn = None
    random_state = None

    def __init__(self, random_state):
        self.random_state = random_state
        self.InitDTLearn()
        self.InitKNNLearn()

    def InitDTLearn(self):
        self.dt_learn = DecisionTreeClassifier(min_samples_split=20,
                                               min_samples_leaf=5,
                                               random_state=self.random_state)
    def InitKNNLearn(self):
        self.knn_learn = KNeighborsClassifier(n_neighbors=3,
                                              algorithm='kd_tree', n_jobs=4)

    def FitDTLearn(self, X, Y):
        self.dt_learn.fit(X, Y)
        return self.dt_learn

    def PredictDTLearn(self, X):
        predictions = self.dt_learn.predict(X)
        return predictions

    def FitKNNLearn(self, X, Y):
        self.knn_learn.fit(X, Y)
        return self.knn_learn

    def PredictKNNLearn(self, X):
        predictions = self.knn_learn.predict(X)
        return predictions