import sys
from time import time

from sklearn.decomposition import NMF
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

class ICAReducer():

    def __init__(self, dataset, dataset_name, num_components=10):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(dataset.data)
        self.n_samples, self.n_features = self.data.shape

        self.reducer = NMF(n_components=num_components, max_iter=5000)

    def reduce(self):
        self.reducer.fit(self.data)
        self.reduced = self.scaler.fit_transform(self.reducer.transform(self.data))
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
        sys.stdout = open('NMFReduceDigitsOutput.txt', 'w')
        print("NMF Reduction of %s:\n" % self.dataset_name)
        print(40 * '-')
        print(self.reduced)
        print("\nLength of 1 input vector before reduction: %d \n" % len(self.data.tolist()[0]))
        print("Length of 1 input vector after reduction: %d \n" % len(self.reduced.tolist()[0]))
        print(40 * '-')
        print(self.reducer.reconstruction_err_)

    def display_reduced_iris(self):
        sys.stdout = open('NMFReduceIrisOutput.txt', 'w')
        print("NMF Reduction of %s:\n" % self.dataset_name)
        print(40 * '-')
        print(self.reduced)
        print("\nLength of 1 input vector before reduction: %d \n" % len(self.data.tolist()[0]))
        print("Length of 1 input vector after reduction: %d \n" % len(self.reduced.tolist()[0]))
        print(40 * '-')
        print(self.reducer.reconstruction_err_)
