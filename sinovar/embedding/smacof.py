"""Stochastic SMACOF embedding of noisy *lower-bound* distances.

The common-line distances written by the ``distance`` command are not distance
estimates in the ordinary sense: a pair of particles is compared along a single
line of their Fourier transforms, so any discrepancy the common line does not
see is missed and the measured value is a *lower bound* of the true distance,
blurred on top of that by shot noise. Classical or landmark MDS fits such data
in the least-squares sense and therefore pulls pairs *together* whenever the
bound is loose, which contracts the embedding towards the pairs whose bound
happens to be weakest.

This module fits the same coordinates against an asymmetric loss that only
penalises embedded distances falling *below* the measurement,

    sigma(X) = sum_{i<j} psi_kappa(delta_ij - d_ij(X)) + lambda sum_i ||x_i||^2

with ``delta_ij = sqrt(D2_ij)`` the measured bound, ``d_ij(X)`` the embedded
distance and ``psi_kappa`` the one-sided Huber function

    psi_kappa(v) = 0                     for v <= 0   (bound met: no force)
                   v^2                   for 0 < v <= kappa
                   2 kappa v - kappa^2   for v > kappa.

The three pieces each handle one defect of the data. The dead zone at ``v <= 0``
makes *downward* noise harmless --- an under-estimated distance is simply a
weaker bound --- so no bias correction of the measurements is needed. The linear
tail beyond ``kappa`` bounds the influence of *upward* noise, which the dead
zone cannot absorb and which would otherwise dominate the fit precisely because
a spuriously large measurement is the most violated one. And the ridge
``lambda`` is what makes the problem well posed at all: the one-sided term alone
is minimised by any configuration large enough to satisfy every bound, so it is
unbounded above and its minimisers form a huge flat set. The ridge selects the
*tightest* configuration consistent with the bounds and shrinks unconstrained
directions to zero instead of letting them float. This is the lower-bound
distance-geometry problem familiar from NMR structure determination, solved by
majorization rather than by embedding.

Majorization carries over from ordinary SMACOF unchanged. For any target
``delta~`` the Guttman majorizer of ``(delta~ - d_ij(X))^2`` at the current
iterate ``Z`` is tight when ``delta~ = delta_ij``, and both the dead zone and
the Huber tail amount to nothing more than *moving the target*:

    delta~_ij = clip(delta_ij, d_ij(Z), d_ij(Z) + kappa)

A satisfied pair targets its current length (zero force, but it keeps its
curvature, which damps the step), a mildly violated pair targets its bound, and
a grossly violated pair targets ``kappa`` beyond where it sits. Substituting
this into the Guttman transform and cancelling the terms a uniform-weight
majorizer contributes to every pair alike leaves a purely repulsive update. For
one point against all others,

    x_p+ = z_p + (F_p(Z) / (N - 1) - ridge * z_p) / (1 + ridge),
    F_p(Z) = sum_{q != p} clip(delta_pq - d_pq, 0, kappa) / d_pq * (z_p - z_q),

where ``ridge = lambda / (N - 1)`` expresses the penalty relative to the pair
curvature, which makes it scale free. This is the exact minimiser of the
majorizer in ``x_p``, so it needs no line search, and it is free of the ``O(N)``
cancellation that forming ``B(Z) Z`` directly would suffer. Every pair
contributes a force of norm at most ``kappa``, so no single measurement can move
a point far however close its partner sits.

Balancing that update radially shows what the ridge buys, and how to choose it:
at a fixed point the mean violation, in units of the RMS distance, is ``ridge``.
It is the *slack* of the fit --- the fraction of a typical distance by which the
bounds are allowed to stay unmet --- so it should be set near the relative noise
of the measurements. Below that the configuration inflates until it clears the
upward noise, above it the configuration contracts and ignores the bounds.

``F_p`` sums over all ``N`` partners of ``p``, i.e. one matrix row, and a full
sweep over every point is the whole ~40 GB matrix --- unaffordable for the tens
of sweeps SMACOF needs. Sweeps are therefore randomized over blocks: each step
draws ``b`` rows without replacement, reads them, and applies the update above
to exactly those ``b`` points using *all* their partners. Every byte read
contributes to the step it was read for, no quantity is estimated from a
subsample, and because each point's update exactly minimises the majorizer in
its own coordinates the objective decreases monotonically --- there is no
sampling noise floor and no step size to tune. One epoch (``N / b`` steps)
visits every point once and so costs exactly one pass over the matrix, but
unlike a deterministic sweep it is Gauss-Seidel rather than Jacobi: batches
later in an epoch already see the points moved by earlier ones. Only the pairs
*internal* to a batch are updated Jacobi-style, and those are a fraction
``b / N`` of each point's partners --- small enough that in practice the batch
size barely changes the trajectory, so it can be chosen purely for throughput.

Peak memory is ``O(b N)`` for the batch of rows and ``O(N k)`` for the
coordinates; the cost is ``n_epochs`` passes over the matrix, which is the knob
to trade against convergence.

The iteration is local, so it needs a global starting point: by default the
landmark MDS solution of :mod:`sinovar.embedding.mds`, which under-estimates the
spread (it fits the bounds rather than respecting them) and therefore starts
inside the violated region, where the update direction is well determined.

One numerical detail drives the whole implementation. The dead zone *rectifies*
error in ``d_ij``: a pair whose distance is over-estimated falls silent, while
one under-estimated by the same amount pushes back, so noise in the distances
does not average out but accumulates as a systematic outward force. The usual
``|z_p|^2 + |z_q|^2 - 2 z_p.z_q`` expansion, which would let the ``(b, N)``
distances be one matrix product, has absolute error ``~eps |z|^2`` and thus
unbounded *relative* error for close pairs --- enough for float32 rounding alone
to inflate an already exact configuration. The distances are therefore formed
from explicit coordinate differences, whose relative error is ``~eps``
regardless of separation, in chunks of columns so that the ``(b, c, k)``
difference tensor stays small.
"""
import logging
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh

from .mds import compute_landmark_mds_embedding

logger = logging.getLogger(__name__)


def _principal_axes(embedding: np.ndarray) -> np.ndarray:
    """Centre an embedding and rotate it onto its principal axes.

    The loss only sees distances, so it fixes the configuration up to a rigid
    motion and leaves the axes in whatever frame the initialization happened to
    use. Rotating them to decreasing spread recovers the ordering classical MDS
    gives for free.

    Going through the ``(k, k)`` Gram matrix rather than an SVD of the ``(N, k)``
    embedding keeps this to one small eigenproblem. It squares the condition
    number, but only an axis whose spread is some ``1e-8`` of the leading one
    suffers, and such an axis has no orientation worth preserving.
    """
    embedding = embedding - embedding.mean(axis=0)

    _, axes = eigh(embedding.T @ embedding)
    rotated = embedding @ axes[:, ::-1]  # eigh ascends; order by spread

    # eigh fixes the eigenvector signs arbitrarily; anchor them on the
    # largest-magnitude coordinate so that repeated runs agree.
    dominant = np.argmax(np.abs(rotated), axis=0)
    signs = np.sign(rotated[dominant, np.arange(rotated.shape[1])])
    return rotated * np.where(signs == 0.0, 1.0, signs)


def _estimate_distance_scale(
    distances2: np.ndarray,
    n_rows: int,
    rng: np.random.Generator
) -> float:
    """RMS measured distance, from a sample of rows.

    Sets the unit for the length-valued quantities (the Huber cut-off, the
    degeneracy guard and the reported slack) so that they can be given relative
    to the data and the whole fit stays equivariant under a rescaling of the
    distances.
    """
    n = distances2.shape[0]
    rows = np.sort(rng.choice(n, size=min(n_rows, n), replace=False))

    # Mean of the squared distances over off-diagonal cells only; the diagonal
    # is zero by construction and would bias the scale downwards for small n.
    total = 0.0
    for row in rows:
        values = np.asarray(distances2[row], dtype=np.float64)  # one memmap row
        total += values.sum() - values[row]

    mean_squared = total / (rows.shape[0] * (n - 1))
    return float(np.sqrt(max(mean_squared, 0.0)))


def _iter_row_batches(
    n: int,
    batch_size: int,
    n_steps: int,
    rng: np.random.Generator
):
    """Yield ``n_steps`` batches of ``batch_size`` distinct row indices.

    Rows are consumed from a permutation, which is redrawn once it no longer
    holds a full batch, so an epoch updates every point once (bar a remainder of
    fewer than ``batch_size`` uniformly random rows). Each batch is sorted, which
    keeps the memmap reads in file order.
    """
    order = rng.permutation(n)
    cursor = 0
    for _ in range(n_steps):
        if cursor + batch_size > order.shape[0]:
            order = rng.permutation(n)
            cursor = 0

        yield np.sort(order[cursor:cursor + batch_size])
        cursor += batch_size


@partial(jax.jit, static_argnames=('column_batch',))
def _smacof_step(
    coords: jax.Array,
    rows: jax.Array,
    batch_distances2: jax.Array,
    kappa: jax.Array,
    ridge: jax.Array,
    step_size: jax.Array,
    distance_floor: jax.Array,
    column_batch: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Majorization step for the ``b`` points whose rows were read.

    ``batch_distances2`` holds the ``(b, N)`` squared distances of those rows.
    Every other point stays put, so this minimises the majorizer over a block of
    coordinates and cannot increase the objective (up to the Jacobi coupling of
    the pairs internal to the batch).
    """
    n, n_components = coords.shape
    batch_size = rows.shape[0]
    batch_coords = coords[rows]                                   # (b, k)

    # The columns are swept in chunks so that nothing of size (b, N) beyond the
    # given distances is ever held.
    force = jnp.zeros((batch_size, n_components), coords.dtype)
    totals = {key: jnp.zeros((), coords.dtype) for key in
              ('stress', 'violated', 'violation')}

    for start in range(0, n, column_batch):
        columns = coords[start:start + column_batch]              # (c, k)
        deltas = jnp.sqrt(jnp.maximum(
            batch_distances2[:, start:start + column_batch], 0.0
        ))                                                        # (b, c) bounds

        # Explicit differences: the inner-product expansion would lose the
        # close pairs, whose error the dead zone below rectifies into a drift.
        diff = batch_coords[:, None, :] - columns[None, :, :]     # (b, c, k)
        d = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1))

        # Huberised one-sided residual: no pull once the bound is met, and a
        # pull capped at kappa so a single over-estimated distance cannot
        # dominate. The floor only guards 0/0 for coincident points; the force
        # it scales vanishes with the separation, so it cannot inflate a step.
        excess = jnp.maximum(deltas - d, 0.0)
        violation = jnp.minimum(excess, kappa)
        weights = violation / jnp.maximum(d, distance_floor)

        # A point is no partner of itself: delta = d = 0 there, which the floor
        # would otherwise turn into a large weight on a null displacement.
        is_self = rows[:, None] == jnp.arange(start, start + columns.shape[0])
        weights = jnp.where(is_self, 0.0, weights)

        # Summing the pair forces themselves rather than contracting the weights
        # against the coordinates: the latter would form F_p as the difference
        # of two sums of size sum_q w_pq * |z|, whose cancellation leaves a
        # relative error the dead zone rectifies into a random walk of the
        # configuration.
        force += jnp.sum(weights[:, :, None] * diff, axis=1)
        totals['stress'] += jnp.sum(violation * (2.0 * excess - violation))
        totals['violated'] += jnp.sum((excess > 0.0) & ~is_self)
        totals['violation'] += jnp.sum(violation)

    direction = (force / (n - 1) - ridge * batch_coords) / (1.0 + ridge)
    coords = coords.at[rows].add(step_size * direction)

    n_pairs = batch_size * (n - 1)
    stats = {
        'stress': totals['stress'] / n_pairs,
        'violated': totals['violated'] / n_pairs,
        # Mean violation in units of the RMS distance: the slack of the fit,
        # which the radial balance of the update drives towards ridge.
        'slack': totals['violation'] / n_pairs,
        'rms_radius': jnp.sqrt(jnp.mean(jnp.sum(jnp.square(coords), axis=-1))),
    }
    return coords, stats


def compute_stochastic_smacof_embedding(
    distances2: np.ndarray,
    n_components: int,
    n_landmarks: int = 4096,
    n_epochs: int = 20,
    batch_rows: int = 256,
    ridge: float = 1e-2,
    huber: float = 0.5,
    step_size: float = 1.0,
    initial: Optional[np.ndarray] = None,
    column_batch: int = 8192,
    random_state: int = 0,
) -> np.ndarray:
    """Embed a squared-distance matrix read as a noisy lower bound.

    Args:
        distances2: symmetric ``(N, N)`` squared-distance matrix, read in row
            batches, so a memmap (``np.load(..., mmap_mode='r')``) works and is
            never materialized in full.
        n_components: number of embedding dimensions. Choose it deliberately:
            unlike classical MDS, which leaves a surplus dimension at its
            near-zero eigenvalue, a one-sided loss can always meet more bounds
            by spreading into whatever dimension it is given, so surplus
            components come back filled rather than empty. The landmark MDS
            eigenvalues logged during initialization are the guide.
        n_landmarks: landmark count for the landmark MDS initialization, ignored
            when ``initial`` is given.
        n_epochs: number of passes over the matrix. Each epoch updates every
            point once, so this is both the iteration count and the I/O budget:
            ``n_epochs`` times the size of the matrix in sequential reads.
        batch_rows: matrix rows read and updated per step. Larger batches use
            the device better but hold ``batch_rows * N`` floats on it, and
            couple more pairs Jacobi-style rather than Gauss-Seidel.
        ridge: compactness penalty relative to the pair curvature
            (``lambda = ridge * (N - 1)``), and the slack of the converged fit:
            the mean violation of the bounds, in units of the RMS distance, is
            driven to this value. Must be positive --- it is what keeps the
            one-sided loss from expanding without bound --- and is best set near
            the relative noise of the measurements. Smaller values inflate the
            embedding until it clears the noise, larger ones contract it and
            ignore the bounds.
        huber: violation at which the loss turns linear, in units of the RMS
            measured distance. Bounds the pull of an over-estimated measurement.
        step_size: relaxation of the majorization step. One is the plain
            Guttman step, which minimises the majorizer exactly and is what the
            monotonicity argument covers; values above it are de Leeuw's
            over-relaxation, which did not pay off in testing, and below it damp
            the step.
        initial: optional ``(N, n_components)`` starting configuration. Defaults
            to the landmark MDS solution.
        column_batch: columns processed at a time within a step, which bounds
            the ``(batch_rows, column_batch, n_components)`` difference tensor.
            Affects device memory only, not the result.
        random_state: seed for the landmark draw and the row sampling.

    Returns:
        The ``(N, n_components)`` embedding, in the same units as the distances
        (not squared), centred on the origin and rotated onto its principal
        axes so the components are ordered by decreasing spread.
    """
    if distances2.ndim != 2 or distances2.shape[0] != distances2.shape[1]:
        raise ValueError('distances2 must be a square matrix')
    if n_components < 1:
        raise ValueError('n_components must be >= 1')
    if n_epochs < 1:
        raise ValueError('n_epochs must be >= 1')
    if batch_rows < 1:
        raise ValueError('batch_rows must be >= 1')
    if column_batch < 1:
        raise ValueError('column_batch must be >= 1')
    if ridge <= 0.0:
        raise ValueError(
            'ridge must be > 0; the one-sided loss is unbounded without it'
        )
    if huber <= 0.0:
        raise ValueError('huber must be > 0')
    if not 0.0 < step_size <= 2.0:
        raise ValueError('step_size must be in (0, 2]')

    n = distances2.shape[0]
    if n < 2:
        raise ValueError('need at least 2 points to embed')

    batch_rows = min(batch_rows, n)
    n_steps = n_epochs * max(1, n // batch_rows)
    rng = np.random.default_rng(random_state)

    scale = _estimate_distance_scale(distances2, 64, rng)
    if not scale > 0.0:
        raise ValueError('the distance matrix is identically zero')

    if initial is None:
        logger.info('Initializing the SMACOF configuration with landmark MDS')
        initial = compute_landmark_mds_embedding(
            distances2,
            n_components=n_components,
            n_landmarks=n_landmarks,
            random_state=random_state,
        )
    elif initial.shape != (n, n_components):
        raise ValueError(
            f'initial must have shape {(n, n_components)}, got {initial.shape}'
        )

    logger.info(
        'Refining %d point(s) over %d epoch(s) of %d step(s) of %d row(s) '
        '(%.1f GB of reads) on %s',
        n, n_epochs, n_steps // n_epochs, batch_rows,
        n_steps * batch_rows * n * 4 / 1e9, jax.devices()[0].device_kind,
    )

    # float32 throughout: jax leaves x64 disabled by default and the matrix
    # itself is float32. What the dead zone cannot tolerate is *relative* error
    # in the distances, which the difference-based step keeps at ~eps.
    coords = jnp.asarray(initial, dtype=jnp.float32)
    kappa = jnp.float32(huber * scale)
    distance_floor = jnp.float32(1e-6 * scale)
    ridge_ = jnp.float32(ridge)
    step_size_ = jnp.float32(step_size)

    steps_per_epoch = n_steps / n_epochs
    log_every = max(1, n_steps // 20)
    for step, rows in enumerate(_iter_row_batches(n, batch_rows, n_steps, rng)):
        batch = np.asarray(distances2[rows], dtype=np.float32)  # (b, N) rows
        coords, stats = _smacof_step(
            coords,
            jnp.asarray(rows),
            jnp.asarray(batch),
            kappa,
            ridge_,
            step_size_,
            distance_floor,
            min(column_batch, n),
        )

        if step % log_every == 0 or step == n_steps - 1:
            logger.info(
                'epoch %5.1f/%d: stress %.4g, %.1f%% of bounds unmet, '
                'slack %.4g (ridge %.4g), radius %.4g',
                (step + 1) / steps_per_epoch, n_epochs, float(stats['stress']),
                100.0 * float(stats['violated']), float(stats['slack']) / scale,
                ridge, float(stats['rms_radius']),
            )

    embedding = _principal_axes(np.asarray(coords, dtype=np.float64))
    logger.info(
        'SMACOF component spread: %s',
        np.array2string(embedding.std(axis=0), precision=3),
    )
    return embedding
