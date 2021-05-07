import numpy as np
from .kmeans import KMeans
from scipy.stats import multivariate_normal


class GMM():
    """"
    A Gaussian Mixture Model updated using expectation maximization.

    Member Variables
    ----------------
    n_clusters (int) - the number of Gaussian clusters\n
    covariance_type (str) - the covariance type for the Gaussians.  Should take the
        value of either \"spherical\" or \"diagonal\"\n
    means (np.ndarray) - the means of the model of shape (# features, n_clusters)\n
    covariances (np.ndarray) - the variances of the model.  If the covariance type is 
        "spherical", then this is of shape (n_clusters), as each feature of a cluster
        has the same variance.  If the covariance type is "diagonal", then this is of
        shape (n_clusters, # features).\n
    mixing_weights (np.ndarray) - the mixing weights of each cluster of shape (n_clusters)\n
    max_iterations (int) - the max number of iterations for the GMM the be optimized. The
        default value is 200.
    """

    def __init__(self, n_clusters: int, covariance_type: str, max_iterations: int = 200):
        self.n_clusters = n_clusters

        if covariance_type in ["spherical", "diagonal"]:
            self.covariance_type = covariance_type
        else:
            raise ValueError(
                "covariance type must equal \"spherical\" or \"diagonal\"")

        self.means = None
        self.covariances = None
        self.mixing_weights = None
        self.max_iterations = max_iterations

    def fit(self, features: np.ndarray) -> None:
        """
        Fit the GMM model

        Parameters
        ----------
        features (np.ndarray) - data of shape (# features, # samples)
        """
        # Initialize means of the GMM with KMeans
        kmeans = KMeans(self.n_clusters)
        kmeans.fit(features)
        self.means = kmeans.means

        # Initialize covariances
        self.covariances = (np.random.rand(self.n_clusters)
                            if self.covariance_type == "spherical"
                            else np.random.rand(self.n_clusters, features.shape[0]))

        # Initialize the mixing weights
        self.mixing_weights = np.random.rand(self.n_clusters)
        self.mixing_weights /= np.sum(self.mixing_weights)

        # Compute log likelihood under initial covariance and means
        prev_log_likelihood = -float("inf")
        log_likelihood = np.sum([self._log_likelihood(features, j)
                                 for j in range(self.n_clusters)])

        # While log_likelihood is increasing significantly or max_iterations has
        # not been reached, continune EM until convergence
        n_iter = 0
        while abs(log_likelihood - prev_log_likelihood) > 1e-4 and n_iter < self.max_iterations:
            prev_log_likelihood = log_likelihood

            assignments = self._e_step(features)
            self.means, self.covariances, self.mixing_weights = self._m_step(
                features, assignments)

            log_likelihood = np.sum([self._log_likelihood(features, j)
                                     for j in range(self.n_clusters)])
            n_iter += 1

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Given features, predict the label of each sample (i.e. the index of the Gaussian
        with the higest posterior for that sample).

        Parameters
        ----------
        features (np.ndarray) - data of shape (# features, # samples)

        Output
        ------
        predictions (np.ndarray) - predicted assignments to each cluster for each sample of
            shape (# samples)
        """
        return np.argmax(self._e_step(features), axis=1)

    def _e_step(self, features: np.ndarray) -> np.ndarray:
        """
        The expectation step in Expectation-Maximization.

        Parameters
        ----------
        features (np.ndarray) - the data of shape (# features, # samples)

        Output
        ------
        Posterior probabilities to each gaussian of shape (# samples, self.n_clusters)
        """
        return np.array([
            self._posterior(features, cluster_idx) for cluster_idx in range(0, self.n_clusters)
        ]).T

    def _m_step(self, features: np.ndarray, assignments: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """"
        The maximization step in Expectation-Maximization.

        Parameters
        ----------
        features (np.ndarray) - features to udpate means of covariances of shape (# features, # samples)\n
        assignments (np.ndarray) - soft assignments of each point to one of the clusters of shape
            (# samples, self.n_clusters)

        Output
        ------
        means (np.ndarray) - updated means of shape (# features, self.n_clusters)\n
        covariances (np.ndarray) - updated covariances of shape (n_clusters, # features or None)\n
        mixing_weights (np.ndarray) - updated mixing weights of shape (n_clusters)
        """
        N = np.sum(assignments, axis=0)  # shape == (# self.n_clusters)

        # Mean Update
        # shape == (# features, self.n_clusters)
        updated_means = np.matmul(features, assignments)/N

        # Covariance Update
        updated_covariances = np.ndarray(
            (self.n_clusters, features.shape[0]))
        for cluster_idx in range(0, self.n_clusters):
            updated_covariances[cluster_idx] = np.sum(assignments[:, cluster_idx].reshape(
                (assignments.shape[0], 1)) * (features.T - updated_means[:, cluster_idx])**2/N[cluster_idx], axis=0)

        # Mixing weights update
        updated_mixing_weights = np.mean(
            assignments, axis=0)  # shape == (n_clusters)

        return (updated_means,
            (updated_covariances if self.covariance_type == "diagonal" else np.mean(updated_covariances, axis=1)),
            updated_mixing_weights)

    def _log_likelihood(self, features: np.ndarray, k_idx: int) -> np.ndarray:
        """
        Compute the log likelihood of the features given the index of the Gaussian 
        in the mixture model.

        Parameters
        ----------
        features (np.ndarray) - the data of shape (# features, # samples)\n
        k_idx (int) - the index of the cluster in the model

        Output
        ------
        log_likelihoods (np.ndarray) - the log likelihoods of each feature for a
            particular Gaussian k_idx of shape (# features, # samples)
        """
        return (np.log(self.mixing_weights[k_idx])
                + multivariate_normal.logpdf(x=features.T,
                                             mean=self.means[:, k_idx],
                                             cov=self.covariances[k_idx])).T

    def _posterior(self, features: np.ndarray, k: int) -> np.ndarray:
        """
        Given the log likelihoods for the GMM and the cluster index, compute the posterior.

        Parameters
        ----------
        features (np.ndarray) - the data of shape (# features, # samples)\n
        k (int) - the index of the cluster in the model

        Output
        ------
        posteriors (np.ndarray) - The posterior probabilities for the selected Gaussian k of
            shape (n_samples)
        """
        numereator = self._log_likelihood(features, k)
        denominator = np.array([
            self._log_likelihood(features, j) for j in range(self.n_clusters)
        ])

        # Logsumexp trick
        max_value = denominator.max(axis=0, keepdims=True)
        denominator_sum = max_value + \
            np.log(np.sum(np.exp(denominator - max_value), axis=0))
        return np.exp(numereator - denominator_sum).flatten()
