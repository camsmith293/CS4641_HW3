import sys
from time import time

from sklearn.random_projection import GaussianRandomProjection
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import numpy as np

class RCAReducer():

    def __init__(self, dataset, dataset_name, num_components=10):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.labels = dataset.target
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(dataset.data)
        self.n_samples, self.n_features = self.data.shape

        self.reducer = GaussianRandomProjection(n_components=num_components)

    def reduce(self):
        self.reducer.fit(self.data)
        self.reduced = self.scaler.fit_transform(self.reducer.transform(self.data))
        return self.reduced

    def benchmark(self, estimator, name, data):
        t0 = time()
        sample_size = 300
        labels = self.labels

        estimator.fit(data)
        print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
              % (name, (time() - t0), estimator.inertia_,
                 metrics.homogeneity_score(labels, estimator.labels_),
                 metrics.completeness_score(labels, estimator.labels_),
                 metrics.v_measure_score(labels, estimator.labels_),
                 metrics.adjusted_rand_score(labels, estimator.labels_),
                 metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
                 metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=sample_size)))

    def display_reduced_digits(self):
        sys.stdout = open('out/RCAReduceDigitsOutput.txt', 'w')
        print("RCA Reduction of %s:\n" % self.dataset_name)
        print(40 * '-')
        print("Length of 1 input vector before reduction: %d \n" % len(self.data.tolist()[0]))
        print("Length of 1 input vector after reduction: %d \n" % len(self.reduced.tolist()[0]))
        print("\nProjection axes:\n")
        for i,axis in enumerate(self.reducer.components_.tolist()):
            print("Axis %d:\n" % i, axis)
        self.compute_plane_variance()

    def compute_plane_variance(self):
        points_along_dimension = self.reduced.T
        for i,points in enumerate(points_along_dimension):
            print("\nVariance of dimension %d:" % i)
            print(np.var(points), "\n")

    def display_reduced_iris(self):
        sys.stdout = open('out/RCAReduceIrisOutput.txt', 'w')
        print("RCA Reduction of %s:\n" % self.dataset_name)
        print(40 * '-')
        print("Length of 1 input vector before reduction: %d \n" % len(self.data.tolist()[0]))
        print("Length of 1 input vector after reduction: %d \n" % len(self.reduced.tolist()[0]))
        print("\nProjection axes:\n")
        for i,axis in enumerate(self.reducer.components_.tolist()):
            print("Axis %d:\n" % i, axis)
        self.compute_plane_variance()

    def reduce_crossvalidation_set(self, X_train, X_test):
        self.reducer.fit(X_train)
        reduced_X_train = self.scaler.transform(X_train)
        reduced_X_test = self.scaler.transform(X_test)
        return reduced_X_train, reduced_X_test