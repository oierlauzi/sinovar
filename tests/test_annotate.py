import numpy as np
import pytest

from sinovar.annotate import (
    GmmPartitioner,
    PcaReducer,
    TruncationReducer,
    UmapReducer,
    build_reducer,
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


def test_pca_reducer_shape_and_variance():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((50, 6))
    reducer = PcaReducer(n_components=2)
    reduced = reducer.reduce(x)
    assert reduced.shape == (50, 2)
    assert reducer.model_ is not None
    assert reducer.model_.explained_variance_ratio_.shape == (2,)


def test_pca_reducer_recovers_dominant_plane():
    # Signal lives in the first two dimensions; the rest is tiny noise.
    rng = np.random.default_rng(3)
    signal = rng.standard_normal((200, 2)) * [10.0, 5.0]
    noise = rng.standard_normal((200, 4)) * 0.01
    x = np.concatenate([signal, noise], axis=1)
    reduced = PcaReducer(n_components=2).reduce(x)
    # The 2D embedding should preserve pairwise structure up to rotation.
    assert reduced.shape == (200, 2)
    assert reduced.std(axis=0).min() > 1.0


def test_pca_reducer_requires_enough_columns():
    with pytest.raises(ValueError):
        PcaReducer(n_components=2).reduce(np.zeros((4, 1)))


def test_build_reducer_dispatch_and_unknown():
    assert isinstance(build_reducer('truncate'), TruncationReducer)
    assert isinstance(build_reducer('pca'), PcaReducer)
    with pytest.raises(ValueError):
        build_reducer('does-not-exist')


def test_umap_reducer():
    pytest.importorskip('umap')
    rng = np.random.default_rng(4)
    x = rng.standard_normal((60, 6))
    reduced = UmapReducer(n_components=2, n_neighbors=10).reduce(x)
    assert reduced.shape == (60, 2)


def test_build_umap_missing_dependency_raises_importerror():
    try:
        import umap  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            build_reducer('umap')
    else:
        pytest.skip('umap-learn is installed; cannot test the missing path')


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
