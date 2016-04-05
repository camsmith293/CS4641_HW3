import sys
from time import time

from sklearn import neural_network, metrics, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetLearner:
    def __init__(self, bunch):
        self.bunch = bunch
        self.counter = 0
        self.X = self.bunch.data
        self.Y = self.bunch.target

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y,
                                                                                test_size=0.3,
                                                                                random_state=0)
        self.scaler = MinMaxScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.fit_transform(self.X_test)

        self.rbm = neural_network.BernoulliRBM(random_state=0, verbose=True)
        self.rbm.learning_rate = 0.06
        self.rbm.n_iter = 100
        self.rbm.n_components = 64

        self.logistic = linear_model.LogisticRegression()
        self.logistic.C = 6000.0
        self.logistic_classifier = linear_model.LogisticRegression(C=100.0)

        self.classifier = Pipeline(steps=[('rbm', self.rbm), ('logistic', self.logistic)])

    def train(self):
        t0 = time()
        self.classifier.fit(self.X_train, self.Y_train)
        self.logistic_classifier.fit(self.X_train, self.Y_train)
        print("Training took ", time() - t0)
        self.counter += 1

    def evaluate(self):
        print("Evaluation of test set using RBM Neural Net:\n%s\n" % (
            metrics.classification_report(
                self.Y_test,
                self.classifier.predict(self.X_test))))

    def plot(self):
        plt.figure(figsize=(4.2, 4))
        for i, comp in enumerate(self.rbm.components_):
            plt.subplot(10, 10, i + 1)
            plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
                       interpolation='nearest')

        plt.xticks(())
        plt.yticks(())
        plt.suptitle('100 components extracted by RBM', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

        plt.savefig("out/NeuralNetLearnerPlot.png")

    def reduce_train(self, reducer):
        outfile = 'out/NeuralNet' + type(reducer).__name__ + 'ReductionOutput.txt'
        sys.stdout = open(outfile, 'w')

        # Pre reduction
        self.train()
        self.evaluate()

        # Reduce
        self.X_train, self.X_test = reducer.reduce_crossvalidation_set(self.X_train, self.X_test)
        self.rbm.n_components = len(self.X_train[0])
        self.train()
        self.evaluate()

    def add_cluster_feature(self, clusterer):
        outfile = 'out/NeuralNet' + type(clusterer).__name__ + 'FeatureOutput.txt'
        sys.stdout = open(outfile, 'w')

        # Pre expansion
        self.train()
        self.evaluate()

        self.X_train = np.append(self.X_train,np.zeros([len(self.X_train),1]),1)
        self.X_test = np.append(self.X_test,np.zeros([len(self.X_test),1]),1)

        clusterer.cluster()

        # Add cluster feature
        for x in self.X_train:
            x[-1] = clusterer.transform(x)

        for x in self.X_test:
            x[-1] = clusterer.transform(x)

        # Pre expansion
        self.train()
        self.evaluate()