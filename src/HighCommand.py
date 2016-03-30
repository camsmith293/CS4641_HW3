from sklearn.datasets import load_digits, load_iris

from kMeansClusterer import kMeansClusterer

digits = load_digits()
iris = load_iris()

digits_name = "Digits Dataset"
iris_name = "Iris Dataset"

def kMeansClusterDigits():
    kMeans = kMeansClusterer(digits, 10, digits_name)
    kMeans.cluster()
    kMeans.display_centroids()

kMeansClusterDigits()