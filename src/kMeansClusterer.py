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

    def cluster(self, iterations=500):
        self.clusterer.max_iter = iterations
        data = scale(self.dataset.data)
        self.clusterer.fit(data)

    def display_digits_centroids(self):
        plt.figure(figsize=(4.2, 4))
        for i, patch in enumerate(self.clusterer.cluster_centers_):
            print(patch, len(patch))
            plt.subplot(10, 10, i + 1)
            plt.imshow(patch.reshape(8, 8), cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())


        plt.suptitle('Centroids of KMeans Clustering of\n ' + self.dataset_name)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

        plt.show()
