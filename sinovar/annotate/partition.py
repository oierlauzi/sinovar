"""Exclusive, exhaustive partitioning of the 2D annotation plane.

A :class:`Partitioner` assigns every particle to exactly one class, so the
resulting annotation is guaranteed to cover all particles without overlap.
Gaussian-mixture partitioning is provided; further strategies (k-means,
polygon rasterisation, watershed, ...) can implement the same interface.
"""
from abc import ABC, abstractmethod
import logging
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

Ellipse = Tuple[np.ndarray, float, float, float]  # (center, width, height, angle_deg)

#: Number of standard deviations drawn for component overlays (~95% region).
ELLIPSE_N_STD = 2.0


class Partitioner(ABC):
    """Assigns each point to exactly one of ``K`` classes."""

    @abstractmethod
    def fit_predict(self, points: np.ndarray) -> np.ndarray:
        """Return an ``(N,)`` integer label array over classes ``0 .. K-1``."""
        raise NotImplementedError


class GmmPartitioner(Partitioner):
    """Gaussian-mixture partitioning by maximum posterior probability.

    ``means_init`` selects the flavour:

    * ``None`` — *automatic*: sklearn initialises the means with k-means.
    * ``(K, 2)`` array — *manual*: the user-placed seeds initialise the means
      (and fix the number of components to ``K``).

    Because every point is assigned to its highest-posterior component, the
    resulting partition is exclusive and exhaustive by construction.
    """

    def __init__(
        self,
        n_components: int,
        means_init: Optional[np.ndarray] = None,
        covariance_type: str = 'full',
        random_state: int = 0,
        max_iter: int = 200,
    ) -> None:
        if n_components < 1:
            raise ValueError('n_components must be >= 1')
        if means_init is not None:
            means_init = np.asarray(means_init, dtype=np.float64)
            n_components = means_init.shape[0]

        self.n_components = n_components
        self.means_init = means_init
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter

        self.model_: Optional[GaussianMixture] = None
        self.labels_: Optional[np.ndarray] = None

    def fit_predict(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            means_init=self.means_init,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        labels = model.fit_predict(points).astype(np.int64)

        self.model_ = model
        self.labels_ = labels
        logger.info(
            'Fitted GMM with %d component(s): converged=%s, BIC=%.4g',
            self.n_components, model.converged_, model.bic(points),
        )
        return labels

    @property
    def n_classes(self) -> int:
        return self.n_components

    def ellipses(self, n_std: float = ELLIPSE_N_STD) -> Iterator[Ellipse]:
        """Yield an ``n_std``-sigma ellipse per component for visualisation."""
        if self.model_ is None:
            return
        for k in range(self.n_components):
            mean = np.asarray(self.model_.means_[k], dtype=np.float64)
            cov = _component_covariance(self.model_, k)
            width, height, angle = _cov_to_ellipse(cov, n_std)
            yield mean, width, height, angle


class ManualSphericalPartitioner(Partitioner):
    """Assign points to fixed, user-specified spherical Gaussian components.

    Unlike :class:`GmmPartitioner`, no EM fitting is performed: the component
    means and isotropic variances are taken as given, mirroring sklearn's
    ``covariance_type='spherical'`` model with frozen parameters. Every point
    is assigned to its highest-posterior component, so the partition is still
    exclusive and exhaustive.
    """

    def __init__(
        self,
        means: np.ndarray,
        variances: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        means = np.asarray(means, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        if means.ndim != 2:
            raise ValueError('means must be a (K, D) array')
        if variances.shape != (means.shape[0],):
            raise ValueError('variances must have one entry per component')
        if np.any(variances <= 0.0):
            raise ValueError('variances must be strictly positive')

        n_components = means.shape[0]
        if weights is None:
            weights = np.full(n_components, 1.0 / n_components)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / weights.sum()

        self.means = means
        self.variances = variances
        self.weights = weights
        self.n_components = n_components
        self.labels_: Optional[np.ndarray] = None

    def fit_predict(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        labels = np.argmax(self._log_responsibilities(points), axis=1)
        labels = labels.astype(np.int64)
        self.labels_ = labels
        logger.info(
            'Assigned points to %d manual spherical component(s)',
            self.n_components,
        )
        return labels

    def predict(self, points: np.ndarray) -> np.ndarray:
        # The model is frozen, so prediction and fit_predict coincide.
        return self.fit_predict(points)

    def _log_responsibilities(self, points: np.ndarray) -> np.ndarray:
        dimension = points.shape[1]
        # Squared distance from every point to every component mean -> (N, K).
        squared_distance = np.sum(
            (points[:, None, :] - self.means[None, :, :]) ** 2, axis=2
        )
        log_density = (
            -0.5 * squared_distance / self.variances[None, :]
            - 0.5 * dimension * np.log(2.0 * np.pi * self.variances[None, :])
        )
        return log_density + np.log(self.weights[None, :])

    @property
    def n_classes(self) -> int:
        return self.n_components

    def ellipses(self, n_std: float = ELLIPSE_N_STD) -> Iterator[Ellipse]:
        """Yield an ``n_std``-sigma circle per component for visualisation."""
        for k in range(self.n_components):
            sigma = float(np.sqrt(self.variances[k]))
            diameter = 2.0 * n_std * sigma
            yield self.means[k], diameter, diameter, 0.0


def fit_gmm_bic(
    points: np.ndarray,
    k_values: Iterable[int],
    **kwargs,
) -> GmmPartitioner:
    """Fit GMMs for each ``k`` in ``k_values`` and keep the lowest-BIC one.

    This is the *automatic* strategy for choosing the number of classes.
    """
    points = np.asarray(points, dtype=np.float64)
    best: Optional[GmmPartitioner] = None
    best_bic = np.inf
    for k in k_values:
        candidate = GmmPartitioner(n_components=k, **kwargs)
        candidate.fit_predict(points)
        bic = candidate.model_.bic(points)
        if bic < best_bic:
            best_bic, best = bic, candidate

    if best is None:
        raise ValueError('k_values must contain at least one value')
    logger.info('Selected K=%d by BIC (%.4g)', best.n_components, best_bic)
    return best


def _component_covariance(model: GaussianMixture, k: int) -> np.ndarray:
    """Return the 2x2 covariance of component ``k`` for any covariance type."""
    covariance_type = model.covariance_type
    covariances = model.covariances_
    if covariance_type == 'full':
        return covariances[k]
    if covariance_type == 'tied':
        return covariances
    if covariance_type == 'diag':
        return np.diag(covariances[k])
    if covariance_type == 'spherical':
        return np.eye(2) * covariances[k]
    raise ValueError(f'Unknown covariance_type: {covariance_type!r}')


def _cov_to_ellipse(cov: np.ndarray, n_std: float) -> Tuple[float, float, float]:
    """Convert a 2x2 covariance to ``(width, height, angle_deg)``."""
    values, vectors = np.linalg.eigh(cov)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(values, 0.0))
    return float(width), float(height), float(angle)
