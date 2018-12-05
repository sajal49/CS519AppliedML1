from sklearn.tree import DecisionTreeClassifier


class SimpleDTLearner:

    dt_learn = None
    min_samples_split = None
    min_samples_leaf = None
    random_state = None

    def __init__(self, random_state, min_samples_split, min_samples_leaf):
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.dt_learn = DecisionTreeClassifier(min_samples_split=self.min_samples_split,
                                               min_samples_leaf=self.min_samples_leaf,
                                               random_state=self.random_state)

    def FitDTLearn(self, X, Y):
        self.dt_learn.fit(X, Y)
        return self.dt_learn

    def PredictDTLearn(self, X):
        predictions = self.dt_learn.predict(X)
        return predictions