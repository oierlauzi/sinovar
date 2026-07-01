"""Dimensionality reduction of embeddings down to the 2D annotation plane.

The annotation GUI operates on a 2D scatter/histogram, so higher-dimensional
embeddings must be reduced first. A :class:`Reducer` maps an ``(N, D)``
embedding to ``(N, n_components)``. Three strategies are provided:

* :class:`TruncationReducer` — keep the leading columns (no dependencies).
* :class:`PcaReducer` — principal component analysis (via scikit-learn).
* :class:`UmapReducer` — UMAP (needs the optional ``umap-learn`` package).

New projections can be added by implementing the :class:`Reducer` interface
and registering them in :data:`REDUCERS`.
"""
from abc import ABC, abstractmethod
import logging

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class Reducer(ABC):
    """Maps an ``(N, D)`` embedding to an ``(N, n_components)`` array."""

    @abstractmethod
    def reduce(self, embedding: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _validate_input(embedding: np.ndarray, n_components: int) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2:
        raise ValueError('embedding must be a 2D array')
    if embedding.shape[1] < n_components:
        raise ValueError(
            f'embedding has {embedding.shape[1]} column(s), '
            f'need at least {n_components}'
        )
    return embedding


class TruncationReducer(Reducer):
    """Keeps the first ``n_components`` columns of the embedding.

    Diffusion-map components are already ordered by decreasing significance,
    so keeping the leading columns is a sensible default.
    """

    def __init__(self, n_components: int = 2) -> None:
        if n_components < 1:
            raise ValueError('n_components must be >= 1')
        self.n_components = n_components

    def reduce(self, embedding: np.ndarray) -> np.ndarray:
        embedding = _validate_input(embedding, self.n_components)
        return embedding[:, :self.n_components]


class PcaReducer(Reducer):
    """Linear projection onto the leading principal components."""

    def __init__(self, n_components: int = 2, random_state: int = 0) -> None:
        if n_components < 1:
            raise ValueError('n_components must be >= 1')
        self.n_components = n_components
        self.random_state = random_state
        self.model_: PCA | None = None

    def reduce(self, embedding: np.ndarray) -> np.ndarray:
        embedding = _validate_input(embedding, self.n_components)
        model = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        reduced = model.fit_transform(embedding)
        self.model_ = model
        logger.info(
            'PCA explained variance ratio: %s',
            np.array2string(model.explained_variance_ratio_, precision=3),
        )
        return reduced


class UmapReducer(Reducer):
    """Non-linear projection with UMAP.

    Requires the optional ``umap-learn`` package (``pip install
    'sinovar[umap]'``); the import is deferred to construction time so that
    the rest of the CLI works without it.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        random_state: int = 0,
    ) -> None:
        if n_components < 1:
            raise ValueError('n_components must be >= 1')
        try:
            import umap  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "UMAP reduction requires the 'umap-learn' package. Install it "
                "with:  pip install 'sinovar[umap]'"
            ) from error

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.model_ = None

    def reduce(self, embedding: np.ndarray) -> np.ndarray:
        import umap

        embedding = _validate_input(embedding, self.n_components)
        model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        reduced = model.fit_transform(embedding)
        self.model_ = model
        return np.asarray(reduced, dtype=np.float64)


#: Reducers selectable by name (e.g. from the CLI).
REDUCERS = {
    'truncate': TruncationReducer,
    'pca': PcaReducer,
    'umap': UmapReducer,
}


def build_reducer(name: str, n_components: int = 2, **kwargs) -> Reducer:
    """Construct a reducer by name.

    Extra keyword arguments are forwarded to the reducer's constructor, so
    strategy-specific options (e.g. UMAP's ``n_neighbors``) can be passed
    through. Raises ``ValueError`` for an unknown name.
    """
    try:
        reducer_cls = REDUCERS[name]
    except KeyError:
        raise ValueError(
            f'Unknown reduction {name!r}; choose from {sorted(REDUCERS)}'
        )
    return reducer_cls(n_components=n_components, **kwargs)
