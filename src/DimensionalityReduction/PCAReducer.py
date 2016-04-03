import sys
from time import time

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import scale


class PCAReducer():

    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.data = scale(dataset.data)
        self.n_samples, self.n_features = self.data.shape

        self.reducer = PCA()

    def reduce(self, num_components=10):
        self.reducer.num_components = num_components
        self.reduced = self.reducer.fit_transform(self.data)
        return self.reduced

    def benchmark(self, estimator, name, data):
        t0 = time()
        labels = data.target
        sample_size = 300

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
        sys.stdout = open('PCAReduceDigitsOutput.txt', 'w')
        print("PCA Reduction of %s:\n" % self.dataset_name)
        print(40 * '-')
        print("Length of 1 input vector before reduction: %d \n" % len(self.data.tolist()[0]))
        print("Length of 1 input vector after reduction: %d \n" % len(self.reduced.tolist()[0]))
        for i,component in enumerate(self.reducer.components_.tolist()):
            print("\nComponent %d: " % i, component)
            print("Variance of component %d :" % i, self.reducer.explained_variance_ratio_[i])

    def display_reduced_iris(self):
        return
