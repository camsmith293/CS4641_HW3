from sklearn.datasets import load_digits, load_iris

from kMeansClusterer import kMeansClusterer
from EMClusterer import EMClusterer

digits = load_digits()
iris = load_iris()

digits_name = "Digits Dataset"
iris_name = "Iris Dataset"

def kMeansClusterDigits():
    kMeans = kMeansClusterer(digits, 10, digits_name)
    kMeans.cluster()
    kMeans.display_digits_centroids()

def EMClusterDigits():
    EM = EMClusterer(digits, 10, digits_name)
    EM.cluster()
    EM.display_digits_centroids()

def kMeansClusterIris():
    kMeans = kMeansClusterer(iris, 3, iris_name)
    kMeans.cluster()
    kMeans.display_iris_clusterings()

def EMClusterIris():
    EM = EMClusterer(iris, 3, iris_name)
    EM.cluster()
    EM.display_iris_clusterings()

kMeansClusterIris()
