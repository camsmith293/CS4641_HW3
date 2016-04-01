from sklearn.datasets import load_digits, load_iris

from kMeansClusterer import kMeansClusterer
from EMClusterer import EMClusterer
from DimensionalityReduction.PCAReducer import PCAReducer
from DimensionalityReduction.ICAReducer import ICAReducer
from DimensionalityReduction.RCAReducer import RCAReducer
from DimensionalityReduction.NMFReducer import NMFReducer

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

def reduce_digits:
    reducers = [PCAReducer(digits, digits_name),
                ICAReducer(digits, digits_name),
                RCAReducer(digits, digits_name),
                NMFReducer(digits, digits_name)]

    for reducer in reducers:
        reducer.reduce()
        reducer.display_reduced_digits()

kMeansClusterIris()
