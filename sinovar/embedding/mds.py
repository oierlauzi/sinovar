"""Classical (metric) MDS embedding of a precomputed squared-distance matrix.

Classical MDS diagonalizes the double-centered squared-distance matrix
``B = -1/2 H D2 H``, which needs the whole matrix in memory and an ``O(N^3)``
eigendecomposition --- hopeless for the ~100k x 100k matrices written by the
``distance`` command (40 GB as float32).

Landmark MDS (de Silva & Tenenbaum, 2004) solves the problem exactly on a
subset of ``m`` landmark points and places every remaining point by
distance-based triangulation against those landmarks. Because the distance
matrix is symmetric, the distances from all points to a landmark are just that
landmark's row, so only ``m`` rows are ever read, and the triangulation is
accumulated one landmark row at a time. Peak memory is ``O(N k + m^2)`` and the
disk traffic ``O(m N)`` --- with ``m = 4096`` and ``N = 100k`` that is ~3 GB of
sequential reads and a few tens of MB of RAM, which makes the whole embedding
I/O bound rather than compute bound. With ``m = N`` the procedure reduces to
plain classical MDS.
"""
import logging

import numpy as np
from scipy.linalg import eigh

logger = logging.getLogger(__name__)


def select_landmarks(
    n: int,
    n_landmarks: int,
    random_state: int = 0
) -> np.ndarray:
    """Pick ``min(n_landmarks, n)`` landmark indices uniformly at random.

    The indices are returned sorted, so the landmark rows are read in file
    order.
    """
    if n_landmarks < 1:
        raise ValueError('n_landmarks must be >= 1')

    m = min(n_landmarks, n)
    if m == n:
        return np.arange(n)

    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=m, replace=False))


def _read_landmark_submatrix(
    distances2: np.ndarray,
    landmarks: np.ndarray
) -> np.ndarray:
    """Gather the ``(m, m)`` landmark-to-landmark squared-distance block."""
    m = landmarks.shape[0]
    submatrix = np.empty((m, m), dtype=np.float64)
    for i, landmark in enumerate(landmarks):
        submatrix[i] = distances2[landmark][landmarks]  # one memmap row

    # Symmetric by construction; average out any asymmetry left by the tiled
    # distance computation so that eigh sees an exactly symmetric input.
    return 0.5 * (submatrix + submatrix.T)


def _classical_mds_basis(
    submatrix: np.ndarray,
    n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Classical MDS of the landmark block.

    Returns the row means of the landmark block (needed to centre the
    triangulation) and the ``(n_components, m)`` triangulation operator
    ``-1/2 Lambda^-1/2 V^T``, i.e. minus half the pseudo-inverse of the
    landmark coordinate matrix.
    """
    m = submatrix.shape[0]

    # B = -1/2 H D2 H with H = I - 11^T/m, expanded so that only the row means
    # and the grand mean are needed.
    row_means = submatrix.mean(axis=1)
    grand_mean = row_means.mean()
    b = -0.5 * (
        submatrix - row_means[:, None] - row_means[None, :] + grand_mean
    )

    # Only the top n_components eigenpairs matter, and asking LAPACK for that
    # subset is far cheaper than a full diagonalization for large m.
    eigvals, eigvecs = eigh(b, subset_by_index=[m - n_components, m - 1])
    # eigh returns ascending eigenvalues; order by decreasing significance.
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # A non-Euclidean (or merely noisy) distance matrix makes B indefinite, and
    # a cloud of intrinsic dimension below n_components leaves eigenvalues at
    # the noise floor. Neither carries embeddable geometry, and dividing by
    # their square root would amplify rounding error into a spurious axis, so
    # they are dropped by zeroing the matching rows of the operator. The
    # threshold is relative to the leading eigenvalue, which sets the scale.
    tolerance = max(eigvals[0], 0.0) * m * np.finfo(eigvals.dtype).eps
    significant = eigvals > tolerance
    if not np.all(significant):
        logger.warning(
            'Dropping %d of %d MDS component(s) with negligible or negative '
            'eigenvalues; the distances span fewer than %d Euclidean dimension(s)',
            int(np.count_nonzero(~significant)), n_components, n_components,
        )

    # eigh fixes the eigenvector signs arbitrarily; anchor them on the
    # largest-magnitude loading so that repeated runs agree.
    dominant = np.argmax(np.abs(eigvecs), axis=0)
    signs = np.sign(eigvecs[dominant, np.arange(eigvecs.shape[1])])
    eigvecs *= np.where(signs == 0.0, 1.0, signs)[None, :]

    scale = np.zeros(eigvals.shape[0])
    scale[significant] = 1.0 / np.sqrt(eigvals[significant])
    operator = -0.5 * (eigvecs * scale[None, :]).T

    logger.info(
        'Landmark MDS eigenvalues: %s',
        np.array2string(eigvals, precision=3),
    )
    return row_means, operator


def compute_landmark_mds_embedding(
    distances2: np.ndarray,
    n_components: int,
    n_landmarks: int = 4096,
    random_state: int = 0,
) -> np.ndarray:
    """Embed every point of a squared-distance matrix with landmark MDS.

    Args:
        distances2: symmetric ``(N, N)`` squared-distance matrix. It is read
            row by row, so a memmap (``np.load(..., mmap_mode='r')``) works and
            is never materialized in full.
        n_components: number of embedding dimensions, ordered by decreasing
            eigenvalue.
        n_landmarks: number of landmarks. Larger values are more accurate but
            cost one extra matrix row of I/O each; ``n_landmarks >= N`` gives
            exact classical MDS.
        random_state: seed for the landmark draw.

    Returns:
        The ``(N, n_components)`` embedding, in the same units as the distances
        (not squared) and centred on the landmark centroid.
    """
    if distances2.ndim != 2 or distances2.shape[0] != distances2.shape[1]:
        raise ValueError('distances2 must be a square matrix')
    if n_components < 1:
        raise ValueError('n_components must be >= 1')

    n = distances2.shape[0]
    landmarks = select_landmarks(n, n_landmarks, random_state=random_state)
    m = landmarks.shape[0]
    if m <= n_components:
        raise ValueError(
            f'need more than n_components={n_components} landmarks, got {m}'
        )

    logger.info('Running classical MDS on %d landmark(s) out of %d', m, n)
    submatrix = _read_landmark_submatrix(distances2, landmarks)
    row_means, operator = _classical_mds_basis(submatrix, n_components)
    del submatrix

    # y_i = -1/2 Lambda^-1/2 V^T (delta_i - mu), where delta_i holds the
    # squared distances from point i to each landmark. Accumulating over
    # landmarks rather than over points turns this into one rank-one update per
    # landmark row, so nothing bigger than the (N, n_components) output and a
    # single matrix row is ever in memory.
    logger.info('Triangulating %d point(s) against the landmarks', n)
    embedding = np.zeros((n, n_components))
    for i, landmark in enumerate(landmarks):
        deltas = np.asarray(distances2[landmark], dtype=np.float64)  # one row
        deltas -= row_means[i]
        embedding += deltas[:, None] * operator[None, :, i]

    return embedding
