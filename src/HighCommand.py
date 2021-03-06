import sys

from sklearn.datasets import load_digits, load_iris

from kMeansClusterer import kMeansClusterer
from EMClusterer import EMClusterer
from NeuralNetworkLearner import NeuralNetLearner
from DimensionalityReduction.PCAReducer import PCAReducer
from DimensionalityReduction.ICAReducer import ICAReducer
from DimensionalityReduction.RCAReducer import RCAReducer
from DimensionalityReduction.NMFReducer import NMFReducer

digits = load_digits()
iris = load_iris()

digits_name = "Digits_Dataset"
iris_name = "Iris_Dataset"

def kMeansClusterDigits():
    sys.stdout = open('out/kMeansClusterDigits.txt', 'w')
    kMeans = kMeansClusterer(digits, 10, digits_name)
    kMeans.cluster()
    kMeans.benchmark("kMeans Digits")
    kMeans.display_digits_centroids()

def EMClusterDigits():
    sys.stdout = open('out/EMClusterDigits.txt', 'w')
    EM = EMClusterer(digits, 10, digits_name)
    EM.cluster()
    EM.benchmark("EM Digits")
    EM.display_digits_centroids()

def kMeansClusterIris():
    sys.stdout = open('out/kMeansClusterIris.txt', 'w')
    kMeans = kMeansClusterer(iris, 3, iris_name)
    kMeans.cluster()
    kMeans.benchmark("kMeans Iris")
    kMeans.display_iris_clusterings()

def EMClusterIris():
    sys.stdout = open('out/EMClusterIris.txt', 'w')
    EM = EMClusterer(iris, 3, iris_name)
    EM.cluster()
    EM.benchmark("EM Iris")
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

def reduce_iris():
    PCAReduce_iris()
    ICAReduce_iris()
    RCAReduce_iris()
    NMFReduce_iris()

def PCAReduce_iris():
    pca = PCAReducer(iris, iris_name, num_components=3)
    pca.reduce()
    pca.display_reduced_iris()

def ICAReduce_iris():
    ica = ICAReducer(iris, iris_name, num_components=3)
    ica.reduce()
    ica.display_reduced_iris()

def RCAReduce_iris():
    rca = RCAReducer(iris, iris_name, num_components=3)
    rca.reduce()
    rca.display_reduced_iris()

def NMFReduce_iris():
    nmf = NMFReducer(iris, iris_name, num_components=3)
    nmf.reduce()
    nmf.display_reduced_iris()

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
    reducers = [PCAReducer(digits, digits_name),
                ICAReducer(digits, digits_name),
                RCAReducer(digits, digits_name),
                NMFReducer(digits, digits_name)]

    for reducer in reducers:
        kMeans = kMeansClusterer(digits, 10, digits_name)
        kMeans.display_reduced_clusterings(reducer)

def kMeansIrisCluster_AllReductions():
    reducers = [PCAReducer(iris, iris_name, num_components=3),
                ICAReducer(iris, iris_name, num_components=3),
                RCAReducer(iris, iris_name, num_components=3),
                NMFReducer(iris, iris_name, num_components=3)]

    for reducer in reducers:
        kMeans = kMeansClusterer(iris, 3, iris_name)
        kMeans.display_reduced_clusterings(reducer)

def EMDigitCluster_AllReductions():
    reducers = [PCAReducer(digits, digits_name),
                ICAReducer(digits, digits_name),
                RCAReducer(digits, digits_name),
                NMFReducer(digits, digits_name)]

    for reducer in reducers:
        EM = EMClusterer(digits, 10, digits_name)
        EM.display_reduced_clusterings(reducer)

def EMIrisCluster_AllReductions():
    reducers = [PCAReducer(iris, iris_name, num_components=3),
                ICAReducer(iris, iris_name, num_components=3),
                RCAReducer(iris, iris_name, num_components=3),
                NMFReducer(iris, iris_name, num_components=3)]

    for reducer in reducers:
        EM = EMClusterer(iris, 3, iris_name)
        EM.display_reduced_clusterings(reducer)

def NeuralNet_PCAReduction():
    pca = PCAReducer(digits, digits_name)
    nnet = NeuralNetLearner(digits)
    nnet.reduce_train(pca)

def clusterFeature_NeuralNet():
    kMeans = kMeansClusterer(digits, 10, digits_name)
    EM = EMClusterer(digits, 10, digits_name)
    nnet = NeuralNetLearner(digits)
    nnet.add_cluster_feature(kMeans, EM)

clusterFeature_NeuralNet()
