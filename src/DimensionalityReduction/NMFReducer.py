from time import time

from sklearn.decomposition import NMF
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


class NMFReducer():

    def __init__(self, dataset, dataset_name, num_components=10):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.data = MinMaxScaler(dataset.data)
        self.n_samples, self.n_features = self.data.shape

        self.reducer = NMF(n_components=num_components)

    def reduce(self):
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
        print("PCA Reduction of %s:\n" % self.dataset_name)

    def display_reduced_iris(self):
        return
