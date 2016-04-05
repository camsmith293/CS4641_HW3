from time import time

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GMM
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import matplotlib.pyplot as plt

class EMClusterer():

    def __init__(self, dataset, num_clusters, dataset_name):
        self.dataset = dataset
        self.num_clusters = num_clusters
        self.dataset_name = dataset_name

        self.scaler = MinMaxScaler()

        self.data = self.scaler.fit_transform(self.dataset.data)
        self.labels = dataset.target

        self.clusterer = GMM(n_components=num_clusters,
                    covariance_type='diag', init_params='wc', n_iter=500, verbose=1)

    def cluster(self, iterations=500):
        for index, (name, clusterer) in enumerate(self.clusterers.items()):
            print("Fitting dataset using covariance type ", clusterer.covariance_type)
            clusterer.n_iter=iterations
            clusterer.fit(self.data)

    def reduce_and_cluster(self, reducer):
        reduced = reducer.reduce()
        self.data = self.scaler.fit_transform(reduced)
        self.cluster()

    def display_digits_centroids(self, outfile='EMDigitsClusterings.png'):
        fig = plt.figure(figsize=(4.2, 4))

        w = 0.08
        h = 0.08
        for j, patch in enumerate(self.clusterer.means_):
            pos = [0.075 + j*1.1*w, 0.18 + 1.2*h, w, h]
            a = fig.add_axes(pos)
            a.imshow(patch.reshape(8, 8), cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())

        plt.suptitle('Centroids of Expected Maximum Clustering of\n ' + self.dataset_name)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

        plt.savefig(outfile)

    def display_iris_clusterings(self, outfile='EMIrisClusterings.png'):
        fig = plt.figure(figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()

        for n, color in enumerate('rgb'):
            data = self.data[self.dataset.target == n]
            ax.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                        label=self.dataset.target_names[n])

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')

        plt.savefig(outfile)

    def benchmark(self, name):
        t0 = time()
        sample_size = 300
        labels = self.labels
        estimator=self.clusterer
        data = self.data

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
