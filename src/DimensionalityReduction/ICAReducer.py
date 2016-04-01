from sklearn.decomposition import FastICA

class ICAReducer():

    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name

        self.data = dataset.data
        self.n_components = len(self.data[0])

        self.reducer = FastICA(n_components=self.n_components)

    def reduce(self):
        return self.reducer.fit_transform(self.data)

    def display_reduced_digits(self):
        return

    def display_reduced_iris(self):
        return