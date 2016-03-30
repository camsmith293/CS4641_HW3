from sklearn.cluster import MiniBatchKMeans

import numpy as np
import matplotlib.pyplot as plt

class kMeansClusterer():

    def __init__(self, dataset, num_clusters, dataset_name):
        self.dataset = dataset
        self.num_clusters = num_clusters
        self.dataset_name = dataset_name

        self.clusterer = MiniBatchKMeans(self.num_clusters, verbose=1)

    def cluster(self, iterations=100):
        self.clusterer.max_iter = iterations
        self.clusterer.fit(self.dataset)

    def display_centroids(self):
        img_size = self.dataset[0].shape

        plt.figure(figsize=(4.2, 4))
        for i, patch in enumerate(self.clusterer.cluster_centers_):
            plt.subplot(9, 9, i + 1)
            plt.imshow(patch.reshape(img_size), cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())


        plt.suptitle('Centroids of Minibatch KMeans Clustering of\n ', self.dataset_name)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

        plt.show()
