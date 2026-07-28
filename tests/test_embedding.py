import numpy as np
import pytest

from sinovar.embedding import (
    compute_landmark_mds_embedding,
    select_landmarks,
)


def _squared_distances(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sum(diff * diff, axis=-1)


def _pairwise_distances(points):
    return np.sqrt(_squared_distances(points))


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
