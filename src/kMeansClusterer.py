import sys

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt

class kMeansClusterer():

    def __init__(self, dataset, num_clusters, dataset_name):
        self.dataset = dataset
        self.num_clusters = num_clusters
        self.dataset_name = dataset_name
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.dataset.data)

        self.clusterer = KMeans(n_clusters=self.num_clusters, verbose=1)

    def cluster(self, iterations=500):
        self.clusterer.max_iter = iterations
        self.clusterer.fit(self.data)

    def reduce_data(self, reducer):
        reduced = reducer.reduce()
        self.data = self.scaler.fit_transform(reduced)
        # self.cluster()

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

    def display_digits_centroids(self, outfile='kMeansDigitsClusterings.png'):
        plt.figure(figsize=(4.2, 4))
        for i, patch in enumerate(self.clusterer.cluster_centers_):
            plt.subplot(10, 10, i + 1)
            plt.imshow(patch.reshape(8, 8), cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())


        plt.suptitle('Centroids of KMeans Clustering of\n ' + self.dataset_name)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

        plt.show()
        plt.savefig(outfile)

    def display_iris_clusterings(self, outfile='kMeansIrisClusterings.png'):
        fig = plt.figure(figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()

        X = self.data
        labels = self.clusterer.labels_

        ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')

        plt.show()
        plt.savefig(outfile)

    def display_reduced_clusterings(self, reducer):
        filename = "out/kMeans" + self.dataset_name + type(reducer).__name__ + "reduction.txt"
        sys.stdout = open(filename, 'w')
        print("%s Reduction and Clustering" % type(reducer).__name__)
        print(40 * '-')
        out_img_pre = 'out/Pre' + type(reducer).__name__ + self.dataset_name + 'kMeans.png'
        self.display_clustering(out_img_pre)
        reducer.benchmark(self, "Pre-Reduction", self.data)
        print(40 * '-')
        self.reduce_data(reducer)
        out_img_pre = 'out/Pre' + type(reducer).__name__ + self.dataset_name + 'kMeans.png'
        self.display_clustering(out_img_pre)
        reducer.benchmark(self, "Post-Reduction", self.data)

