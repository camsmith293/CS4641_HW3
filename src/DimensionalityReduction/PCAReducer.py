from sklearn.decomposition import PCA

class PCAReducer():

    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name

        self.data = dataset.data
        self.n_components = len(self.data[0])

        self.reducer = PCA(n_components=self.n_components)

    def __reduce__(self):
        return self.reducer.fit_transform(self.data)