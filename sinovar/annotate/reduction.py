"""Dimensionality reduction of embeddings down to the 2D annotation plane.

The annotation GUI operates on a 2D scatter/histogram, so higher-dimensional
embeddings must be reduced first. A :class:`Reducer` maps an ``(N, D)``
embedding to ``(N, n_components)``. Only truncation is provided for now;
richer projections (PCA, UMAP, ...) can be added by implementing the same
interface.
"""
from abc import ABC, abstractmethod

import numpy as np


class Reducer(ABC):
    """Maps an ``(N, D)`` embedding to an ``(N, n_components)`` array."""

    @abstractmethod
    def reduce(self, embedding: np.ndarray) -> np.ndarray:
        raise NotImplementedError


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
        embedding = np.asarray(embedding)
        if embedding.ndim != 2:
            raise ValueError('embedding must be a 2D array')
        if embedding.shape[1] < self.n_components:
            raise ValueError(
                f'embedding has {embedding.shape[1]} column(s), '
                f'need at least {self.n_components}'
            )
        return embedding[:, :self.n_components]
