"""Parsing of the ``sinovarEmbedding`` particle column into arrays.

The ``embed`` command stores each particle's embedding as a bracketed,
comma-separated string (e.g. ``"[1.2340e-01, -5.6780e-02]"``). This module
converts that column back into a dense ``(N, D)`` :class:`numpy.ndarray`.
"""
import ast
import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Name of the column written by ``sinovar embed``.
EMBEDDING_COLUMN = 'sinovarEmbedding'

#: Name of the column written by ``sinovar annotate``.
CLASS_COLUMN = 'sinovarClassId'


def parse_embedding_vector(text) -> np.ndarray:
    """Parse a single ``[v0, v1, ...]`` embedding cell into a 1D array."""
    if not isinstance(text, str):
        # Already array-like (e.g. a list or ndarray); normalise to 1D.
        return np.asarray(text, dtype=np.float64).ravel()

    return np.asarray(ast.literal_eval(text), dtype=np.float64).ravel()


def parse_embedding_column(series) -> np.ndarray:
    """Parse the ``sinovarEmbedding`` column into an ``(N, D)`` array.

    Raises:
        ValueError: if the column is empty or particles have inconsistent
            embedding dimensionality.
    """
    rows = [parse_embedding_vector(value) for value in series]
    if not rows:
        raise ValueError('The embedding column is empty')

    lengths = {row.shape[0] for row in rows}
    if len(lengths) != 1:
        raise ValueError(
            'Inconsistent embedding dimensionality across particles: '
            f'{sorted(lengths)}'
        )

    embedding = np.stack(rows, axis=0)
    logger.info(
        'Parsed embedding for %d particles with %d component(s)',
        embedding.shape[0], embedding.shape[1],
    )
    return embedding
