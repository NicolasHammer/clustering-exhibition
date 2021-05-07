import numpy as np


class KMeans():
    """
    The traditional KMeans algorithm with hard assignments.

    Member Variables
    ----------------
    n_clusters (int) - the number of clusters to cluster the data into\n
    means (np.ndarray) - the means of each cluster of shape (# features, n_clusters)
    assignments (np.ndarray) - the assignments of each data point
    """

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.means = None
        self.assignments = None

    def fit(self, features: np.ndarray) -> None:
        """
        Fit the KMeans model to the given data

        Parameters
        ----------
        features (np.ndarray) - data to be trained on of shape (# features, # examples)
        """
        self._initialize_means(features)

        old_assignments = self._update_assignments(features)
        while True:
            self.means = self._update_means(features, old_assignments)
            new_assignments = self._update_assignments(features)

            if np.allclose(old_assignments, new_assignments):
                self.assignments = new_assignments
                return
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

    def _initialize_means(self, features: np.ndarray) -> None:
        """
        Initialize self.means using the k-means++ method
        """
        sample_idxs = np.array(range(0, features.shape[1]))

        previous_idxs = np.array(np.random.choice(sample_idxs)).reshape((1))
        self.means = features[:, previous_idxs[0]
                              ].reshape((features.shape[0], 1))

        for _ in range(1, self.n_clusters):
            unchecked_points = np.delete(features, previous_idxs, axis=1)
            # Compute distances
            distances = np.min(np.sum((
                unchecked_points[:, np.newaxis, :]
                - self.means[:, :, np.newaxis])**2, axis=0), axis=0)

            # Choose next idx
            self.means = np.concatenate((self.means, unchecked_points[:, np.argmax(
                distances)].reshape((features.shape[0], 1))), axis=1)

    def _update_assignments(self, features: np.ndarray) -> np.ndarray:
        """
        Parameters
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
