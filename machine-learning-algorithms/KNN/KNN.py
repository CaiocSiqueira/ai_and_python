import numpy
from collections import Counter

class KNN:

    def __init__(self, k=3):
        self.k = k
    
    def _euclidian(a, b):
        distance = numpy.sqrt(numpy.sum(a-b)**2)
        return distance
    
    def _predict(self, x):
        distances = [self._euclidian(x, self.x_train) for x in self.x_train]

        index_list = numpy.argsort(distances)[:self.k]
        nearest_labels = [self.y_train[i] for i in index_list]

        top_labels = Counter(nearest_labels).most_common()
        return top_labels

    def fit (self, X, y):
        self.x_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

