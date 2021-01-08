import numpy as np


class KMeans():
    """
    The traditional KMeans algorithm with hard assignments.

    Member Variables
    ----------------
    n_clusters (int) - the number of clusters to cluster the data into\n
    means (np.ndarray) - the means of each cluster of shape (# features, n_clusters)
    """

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features: np.ndarray) -> None:
        """
        Fit the KMeans model to the given data

        Parameters
        ----------
        features (np.ndarray) - data to be trained on of shape (# features, # examples)
        """
        init_mean_indices = np.random.permutation(
            range(features.shape[1]))[0:self.n_clusters]
        self.means = features[:, init_mean_indices]

        old_assignments = self._update_assignments(features)

        while True:
            self.means = self._update_means(features, old_assignments)
            new_assignments = self._update_assignments(features)

            if np.allclose(old_assignments, new_assignments):
                break
            else:
                old_assignments = new_assignments

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the cluster of each example in features

        Parameters
        ----------
        features (np.ndarray) - data to predict of shape (# features, # examples) 

        Output
        ------
        predictions (np.ndarray) - clusters labels of each example of shape (# examples)
        """
        return self._update_assignments(features)

    def _update_assignments(self, features: np.ndarray) -> np.ndarray:
        """
        Paramters
        ---------
        features (np.ndarray) - data of shape (# features, # examples)

        Output
        ------
        assignments (np.ndarray) - assignments of each example of shape (# examples)
        """
        return np.argmin(
            np.sum((features[:, :, np.newaxis] -
                    self.means[:, np.newaxis, :])**2, axis=0),
            axis=1)

    def _update_means(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        features (np.ndarray) - data of shape (# features, # examples)\n
        assignments (np.ndarray) - assignments of each example of shape (# examples)

        Output
        ------
        updated_means (np.ndarray) - updated means of shape (# features, self.n_clusters)
        """
        updated_means = np.ndarray((features.shape[0], self.n_clusters))
        for mean_index in range(0, self.n_clusters):
            updated_means[:, mean_index] = np.mean(
                features[:, assignments == mean_index], axis=1)
        return updated_means
