from sklearn.decomposition import PCA

class PCAReducer():

    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name

        self.data = dataset.data
        self.reducer = PCA()

    def reduce(self, num_components=2):
        self.reducer.num_components = num_components
        return self.reducer.fit_transform(self.data)

    def display_reduced_digits(self):
        print("PCA Reduction of %s:\n" % self.dataset_name)

        for i,component in enumerate(self.reducer.components_):
            print("Component %d :" % i, component)
            print("Variance of component %d :", self.reducer.explained_variance_ratio_[i])

    def display_reduced_iris(self):
        return