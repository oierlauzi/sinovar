import numpy as np
import pytest

from sinovar.annotate import (
    GmmPartitioner,
    TruncationReducer,
    fit_gmm_bic,
    parse_embedding_column,
    parse_embedding_vector,
)


def _format_row(row):
    """Mimic the string format written by `sinovar embed`."""
    return '[' + ', '.join(f'{v:.4e}' for v in row) + ']'


def test_parse_embedding_vector_matches_source():
    row = np.array([0.90999, -0.60005, 5.1474e-01])
    parsed = parse_embedding_vector(_format_row(row))
    np.testing.assert_allclose(parsed, row, rtol=1e-4)


def test_parse_embedding_column_shape_and_values():
    original = np.random.default_rng(0).standard_normal((7, 6))
    series = [_format_row(r) for r in original]
    parsed = parse_embedding_column(series)
    assert parsed.shape == (7, 6)
    np.testing.assert_allclose(parsed, original, rtol=1e-4)


def test_parse_embedding_column_rejects_ragged():
    with pytest.raises(ValueError):
        parse_embedding_column(['[1.0, 2.0]', '[1.0, 2.0, 3.0]'])


def test_truncation_reducer_keeps_leading_columns():
    x = np.arange(12.0).reshape(4, 3)
    reduced = TruncationReducer(n_components=2).reduce(x)
    assert reduced.shape == (4, 2)
    np.testing.assert_array_equal(reduced, x[:, :2])


def test_truncation_reducer_requires_enough_columns():
    with pytest.raises(ValueError):
        TruncationReducer(n_components=2).reduce(np.zeros((4, 1)))


def _three_blobs(n=300):
    rng = np.random.default_rng(1)
    centers = np.array([[-6.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    points = np.concatenate([c + rng.standard_normal((n, 2)) for c in centers])
    truth = np.repeat(np.arange(3), n)
    return points, truth


def _agreement(labels, truth):
    """Fraction of points agreeing after best label matching (3 classes)."""
    from itertools import permutations
    best = 0.0
    for perm in permutations(range(3)):
        mapped = np.array(perm)[labels]
        best = max(best, np.mean(mapped == truth))
    return best


def test_gmm_partition_is_exclusive_and_exhaustive():
    points, _ = _three_blobs()
    labels = GmmPartitioner(n_components=3).fit_predict(points)
    assert labels.shape == (points.shape[0],)
    # Every point labelled exactly once within range -> exhaustive & exclusive.
    assert labels.min() >= 0
    assert labels.max() <= 2


def test_gmm_recovers_well_separated_blobs():
    points, truth = _three_blobs()
    labels = GmmPartitioner(n_components=3).fit_predict(points)
    assert _agreement(labels, truth) > 0.95


def test_gmm_manual_seeds_set_component_count():
    points, _ = _three_blobs()
    seeds = np.array([[-6.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    partitioner = GmmPartitioner(n_components=1, means_init=seeds)
    labels = partitioner.fit_predict(points)
    assert partitioner.n_classes == 3
    assert set(np.unique(labels)) == {0, 1, 2}


def test_fit_gmm_bic_selects_three_blobs():
    points, _ = _three_blobs()
    best = fit_gmm_bic(points, range(1, 7))
    assert best.n_components == 3


def test_gmm_ellipses_count_matches_components():
    points, _ = _three_blobs()
    partitioner = GmmPartitioner(n_components=3)
    partitioner.fit_predict(points)
    ellipses = list(partitioner.ellipses())
    assert len(ellipses) == 3
    for mean, width, height, _angle in ellipses:
        assert mean.shape == (2,)
        assert width > 0 and height > 0
