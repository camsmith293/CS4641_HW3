from sklearn.datasets import load_digits, load_iris

from kMeansClusterer import kMeansClusterer
from EMClusterer import EMClusterer
from DimensionalityReduction.PCAReducer import PCAReducer
from DimensionalityReduction.ICAReducer import ICAReducer
from DimensionalityReduction.RCAReducer import RCAReducer
from DimensionalityReduction.NMFReducer import NMFReducer

digits = load_digits()
iris = load_iris()

digits_name = "Digits_Dataset"
iris_name = "Iris_Dataset"

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

def reduce_digits():
    PCAReduce_digits()
    ICAReduce_digits()
    RCAReduce_digits()
    NMFReduce_digits()

def PCAReduce_digits():
    pca = PCAReducer(digits, digits_name)
    pca.reduce()
    pca.display_reduced_digits()

def ICAReduce_digits():
    ica = ICAReducer(digits, digits_name)
    ica.reduce()
    ica.display_reduced_digits()

def RCAReduce_digits():
    rca = RCAReducer(digits, digits_name)
    rca.reduce()
    rca.display_reduced_digits()

def NMFReduce_digits():
    nmf = NMFReducer(digits, digits_name)
    nmf.reduce()
    nmf.display_reduced_digits()

def kMeansDigitCluster_PCAReduce():
    kMeans = kMeansClusterer(digits, 10, digits_name)
    pca = PCAReducer(digits, digits_name)
    kMeans.display_reduced_clusterings(pca)

def kMeansDigitCluster_ICAReduce():
    kMeans = kMeansClusterer(digits, 10, digits_name)
    ica = PCAReducer(digits, digits_name)
    kMeans.display_reduced_clusterings(ica)

def kMeansDigitCluster_RCAReduce():
    kMeans = kMeansClusterer(digits, 10, digits_name)
    rca = RCAReducer(digits, digits_name)
    kMeans.display_reduced_clusterings(rca)

def kMeansDigitCluster_NMFReduce():
    kMeans = kMeansClusterer(digits, 10, digits_name)
    nmf = NMFReducer(digits, digits_name)
    kMeans.display_reduced_clusterings(nmf)

def kMeansDigitCluster_AllReductions():
    kMeans = kMeansClusterer(digits, 10, digits_name)
    pca = PCAReducer(digits, digits_name)
    ica = ICAReducer(digits, digits_name)
    rca = RCAReducer(digits, digits_name)
    nmf = NMFReducer(digits, digits_name)
    kMeans.display_reduced_clusterings(pca)
    kMeans.display_reduced_clusterings(ica)
    kMeans.display_reduced_clusterings(rca)
    kMeans.display_reduced_clusterings(nmf)


kMeansDigitCluster_AllReductions()
