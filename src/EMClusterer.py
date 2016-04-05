from time import time

import sys
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GMM
from sklearn.preprocessing import MinMaxScaler

import numpy as np
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
        self.clusterer.n_iter=iterations
        self.clusterer.fit(self.data)

    def reduce_data(self, reducer):
        reduced = reducer.reduce()
        self.data = self.scaler.fit_transform(reduced)

    def display_clustering(self, outfile=None):
        if self.dataset_name is "Digits_Dataset":
            if outfile is None:
                self.display_digits_centroids()
            else:
                self.display_digits_centroids(outfile)
        else:
            if outfile is None:
                self.display_iris_clusterings()
            else:
                self.display_iris_clusterings(outfile)

    def display_digits_centroids(self, outfile='out/EMDigitsClusterings.png'):
        plt.figure(figsize=(4.2, 4))
        for i, patch in enumerate(self.clusterer.means_):
            plt.subplot(10, 10, i + 1)
            plt.imshow(patch.reshape(8, 8), cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())

        plt.suptitle('Centroids of EM Clustering of\n ' + self.dataset_name)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

        plt.savefig(outfile)

    def display_iris_clusterings(self, outfile='out/EMIrisClusterings.png'):
        fig = plt.figure(figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()

        X = self.data
        labels = self.clusterer.predict(X)
        max_dim = len(X[0]) - 1
        ax.scatter(X[:, max_dim], X[:, 0], X[:, 2], c=labels.astype(np.float))

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')

        plt.savefig(outfile)

    def display_reduced_clusterings(self, reducer):
        filename = "out/EM" + self.dataset_name + type(reducer).__name__ + "reduction.txt"
        sys.stdout = open(filename, 'w')
        print("%s Reduction and Clustering" % type(reducer).__name__)
        print(40 * '-')
        out_img_pre = 'out/Pre' + type(reducer).__name__ + self.dataset_name + 'EM.png'
        self.cluster()
        self.display_clustering(out_img_pre)
        self.benchmark("Pre-Reduction")
        print(40 * '-')
        self.reduce_data(reducer)
        self.clusterer = GMM(n_components=self.num_clusters,
                    covariance_type='diag', init_params='wc', n_iter=500, verbose=1)
        self.cluster()
        out_img_post = 'out/Post' + type(reducer).__name__ + self.dataset_name + 'EM.png'
        self.display_clustering(out_img_post)
        self.benchmark("Post-Reduction")

    def benchmark(self, name):
        self.clusterer.fit(self.data)
        print("AIC Score:%d" % self.clusterer.aic(self.data))
        print("BIC Score:%d" % self.clusterer.bic(self.data))

    def append_with_clustering(self):
        self.cluster()
        labels = self.clusterer.predict(self.data)
        appended = np.append(self.data, np.zeros([len(self.data),1]),1)
        for i,x in enumerate(appended):
            x[-1] = labels[i]
        return appended