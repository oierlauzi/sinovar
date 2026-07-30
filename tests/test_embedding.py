import numpy as np
import pytest

from sinovar.embedding import (
    compute_landmark_mds_embedding,
    compute_stochastic_smacof_embedding,
    select_landmarks,
)


def _squared_distances(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sum(diff * diff, axis=-1)


def _pairwise_distances(points):
    return np.sqrt(_squared_distances(points))


def _distance_error(embedding, points):
    """RMS pairwise-distance error, relative to the RMS distance."""
    got = _pairwise_distances(embedding)
    want = _pairwise_distances(points)
    return np.sqrt(np.mean((got - want) ** 2)) / np.sqrt(np.mean(want ** 2))


def _loosen(distances2, fraction, floor, random_state):
    """Weaken a fraction of the pairs into strictly looser lower bounds.

    Multiplies the *distances* of a random symmetric subset of pairs by a factor
    in ``[floor, 1)``, which is what the common-line measurement does when its
    single line misses part of the discrepancy.
    """
    rng = np.random.default_rng(random_state)
    n = distances2.shape[0]

    factor = rng.uniform(floor, 1.0, size=(n, n))
    loosen = rng.random((n, n)) < fraction
    # Symmetrise both from the upper triangle, leaving the diagonal untouched.
    factor = np.triu(factor, 1)
    factor = factor + factor.T + np.eye(n)
    loosen = np.triu(loosen, 1)
    loosen = loosen | loosen.T

    return distances2 * np.where(loosen, factor, 1.0) ** 2


def test_select_landmarks_is_sorted_and_unique():
    landmarks = select_landmarks(100, 10, random_state=3)
    assert landmarks.shape == (10,)
    assert np.all(np.diff(landmarks) > 0)
    assert landmarks.min() >= 0 and landmarks.max() < 100


def test_select_landmarks_takes_everything_when_asked_for_more():
    np.testing.assert_array_equal(select_landmarks(5, 50), np.arange(5))


def test_select_landmarks_is_deterministic():
    np.testing.assert_array_equal(
        select_landmarks(1000, 32, random_state=7),
        select_landmarks(1000, 32, random_state=7),
    )


def test_exact_mds_recovers_euclidean_geometry():
    """With every point a landmark, classical MDS is exact up to a rigid motion."""
    points = np.random.default_rng(0).standard_normal((60, 3))
    distances2 = _squared_distances(points)

    y = compute_landmark_mds_embedding(distances2, n_components=3, n_landmarks=60)

    assert y.shape == (60, 3)
    np.testing.assert_allclose(
        _pairwise_distances(y), _pairwise_distances(points), atol=1e-8
    )


def test_landmark_mds_recovers_geometry_from_a_subset():
    """A landmark subset spanning the space is enough to place every point."""
    points = np.random.default_rng(1).standard_normal((300, 2))
    distances2 = _squared_distances(points)

    y = compute_landmark_mds_embedding(distances2, n_components=2, n_landmarks=40)

    assert y.shape == (300, 2)
    np.testing.assert_allclose(
        _pairwise_distances(y), _pairwise_distances(points), atol=1e-6
    )


def test_components_are_ordered_by_decreasing_spread():
    rng = np.random.default_rng(2)
    points = rng.standard_normal((200, 3)) * np.array([10.0, 3.0, 1.0])
    distances2 = _squared_distances(points)

    y = compute_landmark_mds_embedding(distances2, n_components=3, n_landmarks=50)

    spread = y.std(axis=0)
    assert np.all(np.diff(spread) < 0)


def test_embedding_is_centred_on_the_landmark_centroid():
    points = np.random.default_rng(3).standard_normal((80, 3)) + 5.0
    distances2 = _squared_distances(points)
    landmarks = select_landmarks(80, 30, random_state=0)

    y = compute_landmark_mds_embedding(distances2, n_components=3, n_landmarks=30)

    np.testing.assert_allclose(y[landmarks].mean(axis=0), 0.0, atol=1e-8)


def test_extra_components_of_a_degenerate_cloud_are_dropped():
    """Points on a plane have no third dimension to embed."""
    rng = np.random.default_rng(4)
    points = np.column_stack((rng.standard_normal((50, 2)), np.zeros(50)))
    distances2 = _squared_distances(points)

    y = compute_landmark_mds_embedding(distances2, n_components=3, n_landmarks=50)

    np.testing.assert_allclose(y[:, 2], 0.0, atol=1e-8)
    np.testing.assert_allclose(
        _pairwise_distances(y), _pairwise_distances(points), atol=1e-8
    )


def test_reads_a_memmapped_matrix_row_by_row(tmp_path):
    """The 100k x 100k case never fits in RAM, so memmaps must work."""
    points = np.random.default_rng(5).standard_normal((120, 2))
    path = tmp_path / 'distances2.npy'
    np.save(path, _squared_distances(points).astype(np.float32))

    distances2 = np.load(path, mmap_mode='r')
    y = compute_landmark_mds_embedding(distances2, n_components=2, n_landmarks=40)

    np.testing.assert_allclose(
        _pairwise_distances(y), _pairwise_distances(points), rtol=1e-3, atol=1e-3
    )


def test_rejects_a_non_square_matrix():
    with pytest.raises(ValueError):
        compute_landmark_mds_embedding(np.zeros((4, 5)), n_components=2)


def test_rejects_fewer_landmarks_than_components():
    distances2 = _squared_distances(np.random.default_rng(6).standard_normal((20, 3)))
    with pytest.raises(ValueError):
        compute_landmark_mds_embedding(distances2, n_components=3, n_landmarks=3)


def test_landmark_mds_leaves_the_distance_matrix_alone():
    """The triangulation centres rows in place, so it must copy them first."""
    distances2 = _squared_distances(np.random.default_rng(7).standard_normal((40, 2)))
    original = distances2.copy()

    compute_landmark_mds_embedding(distances2, n_components=2, n_landmarks=20)

    np.testing.assert_array_equal(distances2, original)


def test_smacof_recovers_geometry_from_tight_bounds():
    """With the bounds met exactly, the truth is very nearly the fixed point."""
    points = np.random.default_rng(0).standard_normal((120, 2)) * [4.0, 1.0]
    distances2 = _squared_distances(points)

    y = compute_stochastic_smacof_embedding(
        distances2, n_components=2, n_landmarks=40, ridge=1e-3
    )

    assert y.shape == (120, 2)
    # The ridge trades a little shrinkage for well-posedness, so this is close
    # rather than exact.
    assert _distance_error(y, points) < 0.02


def test_smacof_beats_least_squares_mds_on_lower_bounds():
    """The whole point: least squares chases loose bounds, the dead zone ignores them."""
    points = np.random.default_rng(1).standard_normal((200, 2)) * [4.0, 1.0]
    distances2 = _squared_distances(points)
    bounds2 = _loosen(distances2, fraction=0.6, floor=0.4, random_state=2)

    mds = compute_landmark_mds_embedding(bounds2, n_components=2, n_landmarks=64)
    smacof = compute_stochastic_smacof_embedding(
        bounds2, n_components=2, n_landmarks=64
    )

    assert _distance_error(smacof, points) < 0.6 * _distance_error(mds, points)


def test_smacof_starting_from_the_answer_stays_there():
    """A correct configuration must be a fixed point, not merely a good start."""
    points = np.random.default_rng(3).standard_normal((150, 2)) * [4.0, 1.0]
    points -= points.mean(axis=0)
    bounds2 = _loosen(_squared_distances(points), 0.5, 0.5, random_state=4)

    y = compute_stochastic_smacof_embedding(
        bounds2, n_components=2, n_epochs=60, ridge=1e-3, initial=points
    )

    assert _distance_error(y, points) < 0.02


def test_smacof_drives_the_slack_towards_the_ridge():
    """Balancing the update radially predicts mean violation / RMS distance = ridge."""
    points = np.random.default_rng(5).standard_normal((150, 2)) * [4.0, 1.0]
    bounds2 = _loosen(_squared_distances(points), 0.5, 0.5, random_state=6)
    ridge = 3e-2

    y = compute_stochastic_smacof_embedding(
        bounds2, n_components=2, n_landmarks=64, n_epochs=120, ridge=ridge
    )

    unmet = np.maximum(np.sqrt(bounds2) - _pairwise_distances(y), 0.0)
    np.fill_diagonal(unmet, 0.0)
    n = points.shape[0]
    n_pairs = n * (n - 1)
    scale = np.sqrt(bounds2.sum() / n_pairs)      # RMS measured distance
    slack = unmet.sum() / n_pairs / scale

    assert 0.3 * ridge < slack < 1.5 * ridge


def test_smacof_is_bounded_when_the_bounds_say_nothing():
    """Vacuous bounds leave only the ridge, which must not let anything run away."""
    n = 80
    bounds2 = np.full((n, n), 1e-8)
    np.fill_diagonal(bounds2, 0.0)

    y = compute_stochastic_smacof_embedding(bounds2, n_components=2, n_landmarks=40)

    assert np.all(np.isfinite(y))
    assert np.sqrt(np.mean(np.sum(y * y, axis=-1))) < 1e-3


def test_smacof_caps_the_pull_of_over_estimated_distances():
    """A few distances far above the truth must not tear the embedding apart."""
    rng = np.random.default_rng(7)
    points = rng.standard_normal((150, 2)) * [4.0, 1.0]
    distances2 = _squared_distances(points)

    outlier = rng.random((150, 150)) < 0.02
    outlier = np.triu(outlier, 1)
    outlier = outlier | outlier.T
    corrupted = np.where(outlier, distances2 * 25.0, distances2)

    capped = compute_stochastic_smacof_embedding(
        corrupted, n_components=2, n_landmarks=64, huber=0.2
    )
    uncapped = compute_stochastic_smacof_embedding(
        corrupted, n_components=2, n_landmarks=64, huber=1e6
    )

    assert _distance_error(capped, points) < _distance_error(uncapped, points)


def test_smacof_components_are_ordered_by_decreasing_spread():
    points = np.random.default_rng(8).standard_normal((150, 3)) * [10.0, 3.0, 1.0]
    distances2 = _squared_distances(points)

    y = compute_stochastic_smacof_embedding(
        distances2, n_components=3, n_landmarks=64
    )

    assert np.all(np.diff(y.std(axis=0)) < 0)


def test_smacof_is_centred_and_deterministic():
    points = np.random.default_rng(9).standard_normal((100, 2)) + 5.0
    distances2 = _squared_distances(points)

    kwargs = dict(n_components=2, n_landmarks=40, n_epochs=5, random_state=11)
    y = compute_stochastic_smacof_embedding(distances2, **kwargs)

    np.testing.assert_allclose(y.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_array_equal(
        y, compute_stochastic_smacof_embedding(distances2, **kwargs)
    )


def test_smacof_reads_a_memmapped_matrix_in_row_batches(tmp_path):
    """The 100k x 100k case never fits in RAM, so memmaps must work."""
    points = np.random.default_rng(10).standard_normal((120, 2)) * [4.0, 1.0]
    path = tmp_path / 'distances2.npy'
    np.save(path, _squared_distances(points).astype(np.float32))

    distances2 = np.load(path, mmap_mode='r')
    y = compute_stochastic_smacof_embedding(
        distances2, n_components=2, n_landmarks=40, batch_rows=16, ridge=1e-3
    )

    assert _distance_error(y, points) < 0.05


def test_smacof_column_batching_does_not_change_the_result():
    points = np.random.default_rng(12).standard_normal((90, 2)) * [4.0, 1.0]
    distances2 = _squared_distances(points)

    kwargs = dict(n_components=2, n_landmarks=40, n_epochs=5)
    whole = compute_stochastic_smacof_embedding(
        distances2, column_batch=1024, **kwargs
    )
    chunked = compute_stochastic_smacof_embedding(
        distances2, column_batch=16, **kwargs
    )

    np.testing.assert_allclose(whole, chunked, atol=1e-4)


@pytest.mark.parametrize('kwargs', [
    dict(n_components=0),
    dict(n_components=2, n_epochs=0),
    dict(n_components=2, batch_rows=0),
    dict(n_components=2, column_batch=0),
    dict(n_components=2, ridge=0.0),
    dict(n_components=2, huber=0.0),
    dict(n_components=2, step_size=0.0),
    dict(n_components=2, step_size=2.5),
])
def test_smacof_rejects_out_of_range_arguments(kwargs):
    distances2 = _squared_distances(np.random.default_rng(13).standard_normal((20, 2)))
    with pytest.raises(ValueError):
        compute_stochastic_smacof_embedding(distances2, **kwargs)


def test_smacof_rejects_a_non_square_matrix():
    with pytest.raises(ValueError):
        compute_stochastic_smacof_embedding(np.zeros((4, 5)), n_components=2)


def test_smacof_rejects_a_mismatched_initial_configuration():
    distances2 = _squared_distances(np.random.default_rng(14).standard_normal((20, 2)))
    with pytest.raises(ValueError):
        compute_stochastic_smacof_embedding(
            distances2, n_components=2, initial=np.zeros((20, 3))
        )
