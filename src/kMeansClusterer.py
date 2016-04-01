from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

import numpy as np
import matplotlib.pyplot as plt

class kMeansClusterer():

    def __init__(self, dataset, num_clusters, dataset_name):
        self.dataset = dataset
        self.num_clusters = num_clusters
        self.dataset_name = dataset_name

        self.clusterer = KMeans(self.num_clusters, verbose=1)

    def cluster(self, iterations=500, data = None):
        self.clusterer.max_iter = iterations
        #data = scale(self.dataset.data)
        if data is None:
            data = self.dataset.data
        self.clusterer.fit(data)

    def display_digits_centroids(self):
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

    def display_iris_clusterings(self):
        fig = plt.figure(figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()

        X = self.dataset.data
        labels = self.clusterer.labels_

        ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')

        plt.show()
