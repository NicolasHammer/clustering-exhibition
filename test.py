from sklearn.datasets import make_blobs
from from_scratch.kmeans import KMeans
from from_scratch.gmm import GMM
import numpy as np
import os
import random
from scipy.stats import multivariate_normal
from itertools import permutations
from sklearn.metrics import adjusted_mutual_info_score as adjusted_mutual_info

np.random.seed(0)
random.seed(0)

# Test KMeans on Generated
n_samples = [1000, 10000]
n_centers = [2]
stds = [.1]
n_features = [1, 2, 4]

for n in n_samples:
    for f in n_features:
        for c in n_centers:
            for s in stds:
                features, targets = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.0)

                # make model and fit
                model = KMeans(c)
                model.fit(features.T)

                means = model.means.T
                orderings = permutations(means)
                distance_to_true_means = []

                actual_means = np.array([
                    features[targets == i, :].mean(axis=0) for i in range(targets.max() + 1)
                ])
                
                for ordering in orderings:
                    _means = np.array(list(ordering))
                    
                    distance_to_true_means.append(
                        np.abs(_means - actual_means).sum()
                    )
                
                assert (min(distance_to_true_means) < 1e-1)

                # predict and calculate adjusted mutual info
                labels = model.predict(features.T)
                acc = adjusted_mutual_info(targets, labels)
                assert (acc >= .9)


# Test GMM Spec
features, targets = make_blobs(n_samples=100, n_features=2, n_centers=2, cluster_stds=.1)
gmm = GMM(2, 'spherical')
gmm.fit(features.T)

assert (hasattr(gmm, 'means'))
assert (hasattr(gmm, 'covariances'))
assert (hasattr(gmm, 'mixing_weights'))

# Test KMeans Spec
features, targets = make_blobs(n_samples=100, n_features=2, n_centers=2, cluster_stds=.1)
model = KMeans(2)
model.fit(features.T)
assert (hasattr(model, 'means'))

# Test GMM Likelihood
features = np.random.rand(4, 2)
means = np.random.rand(2, 2)
covariances = np.random.rand(2, 2)
mixing_weights = np.array([1, 1])

gmm = GMM(means.shape[0], 'diagonal')
gmm.means = means
gmm.covariances = covariances
gmm.mixing_weights = mixing_weights

for k in range(means.shape[0]):
    scipy_prob = multivariate_normal.logpdf(
        features, means[k], covariances[k]
    )
    gmm_prob = gmm._log_likelihood(features.T, k)
    assert np.allclose(scipy_prob, gmm_prob)

# Test GMM Parameters
n_samples = [1000]
n_centers = [2]
stds = [.1, .5]
n_features = [2, 4]

for n in n_samples:
    for f in n_features:
        for c in n_centers:
            for s in stds:
                features, targets = make_blobs(
                    n_samples=n,
                    n_features=f,
                    n_centers=c,
                    cluster_stds=s
                )
                # make model and fit
                model = GMM(c, covariance_type="spherical")
                model.fit(features.T)
                covariances = model.covariances
                for cov in covariances:
                    assert (np.abs(np.sqrt(cov) - s).mean() < 1e-1)

                means = model.means
                orderings = permutations(means)
                distance_to_true_means = []

                actual_means = np.array([
                    features[targets == i, :].mean(axis=0) for i in range(targets.max() + 1)
                ])

                for ordering in orderings:
                    _means = np.array(list(ordering))
                    
                    distance_to_true_means.append(
                        np.abs(_means - actual_means).sum()
                    )
                assert (min(distance_to_true_means) < 1e-1)

                mixing_weights = model.mixing_weights
                orderings = permutations(mixing_weights)
                distance_to_true_mixing_weights = []

                actual_mixing_weights = np.array([
                    features[targets == i, :].shape[0] for i in range(targets.max() + 1)
                ])
                actual_mixing_weights = actual_mixing_weights / actual_mixing_weights.sum()

                for ordering in orderings:
                    _mixing_weights = np.array(list(ordering))
                    
                    distance_to_true_mixing_weights.append(
                        np.abs(_mixing_weights - actual_mixing_weights).sum()
                    )
                assert (min(distance_to_true_mixing_weights) < 1e-1)

                # predict and calculate adjusted mutual info
                labels = model.predict(features.T)
                acc = adjusted_mutual_info(targets, labels)
                assert (acc >= .9)