from sklearn.random_projection import GaussianRandomProjection
from sklearn.datasets import load_iris

class RCAReducer():

    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name

        self.data = dataset.data
        self.n_components = len(self.data[0])

        self.reducer = GaussianRandomProjection(n_components=self.n_components)

    def reduce(self):
        return self.reducer.fit_transform(self.data)

rca = RCAReducer(load_iris(), "iris")
print(rca.reduce())