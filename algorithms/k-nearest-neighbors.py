import numpy
from collections import Counter

def distance(x1, x2):
    return numpy.sqrt(numpy.sum(x1-x2)**2)

class KNN:

    def __init__(self, k=3):
        self.k = k    
   
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted = [self._predicted(x) for x in X]
        return numpy.array(predicted)

    def _predict(self, x):
        distances = [distance(x, x_train) for x_train in self.X_train]

        k_indices = numpy.argsort(distances)[:self.k]
        k_nearest_labes = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labes).most_common(1)
        return most_common[0][0]
