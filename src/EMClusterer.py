from matplotlib import cm
from sklearn.mixture import GMM
from sklearn.preprocessing import scale

import numpy as np
import matplotlib.pyplot as plt

class EMClusterer():

    def __init__(self, dataset, num_clusters, dataset_name):
        self.dataset = dataset
        self.num_clusters = num_clusters
        self.dataset_name = dataset_name

        self.clusterers = dict((covar_type, GMM(n_components=num_clusters,
                    covariance_type=covar_type, init_params='wc', n_iter=500))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])

    def cluster(self, iterations=500):
        data = scale(self.dataset.data)
        for index, (name, clusterer) in enumerate(self.clusterers.items()):
            clusterer.n_iter=iterations
            clusterer.fit(data)


    def display_digits_centroids(self):
        plt.figure(figsize=(4.2, 4))
        for index, (name, clusterer) in enumerate(self.clusterers.items()):
            for i, patch in enumerate(clusterer.means_):
                print(patch, len(patch))
                plt.subplot(10, 10, i + index + 1)
                plt.imshow(patch.reshape(8, 8), cmap=plt.cm.gray,
                           interpolation='nearest')
                plt.xticks(())
                plt.yticks(())

        # colors = cm.rainbow(np.linspace(0, 1, 10))
        #
        # for n, color in enumerate(colors):
        #     data = self.dataset.data[self.dataset.target == n]
        #     plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
        #                 label=self.dataset.target_names[n])


        plt.suptitle('Centroids of Expected Maximum Clustering of\n ' + self.dataset_name)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

        plt.show()
