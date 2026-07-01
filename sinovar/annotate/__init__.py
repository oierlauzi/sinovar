"""Interactive annotation of the sinovar embedding into particle classes.

Only the lightweight, dependency-safe pieces are exported here. The
matplotlib-based GUI lives in :mod:`sinovar.annotate.gui` and must be imported
explicitly, so that this package can be used (and the ``sinovar`` CLI can load)
without the optional ``annotate`` dependencies installed.
"""
from .embedding import (
    CLASS_COLUMN,
    EMBEDDING_COLUMN,
    parse_embedding_column,
    parse_embedding_vector,
)
from .partition import GmmPartitioner, Partitioner, fit_gmm_bic
from .reduction import Reducer, TruncationReducer

__all__ = [
    'CLASS_COLUMN',
    'EMBEDDING_COLUMN',
    'parse_embedding_column',
    'parse_embedding_vector',
    'GmmPartitioner',
    'Partitioner',
    'fit_gmm_bic',
    'Reducer',
    'TruncationReducer',
]
