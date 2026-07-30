"""Stochastic refinement of an existing embedding against common lines.

Where :mod:`sinovar.embedding` *builds* an embedding out of measured distances,
this refines one that already exists --- a RECOVAR embedding, say --- against the
same measurements, but with the measurement's own defects folded into the model
instead of being treated as noise on an ideal distance. Given the eigen-volumes
``U`` that go with the embedding, the distance a pair of particles *should*
exhibit along its common line is computable: it is the common line of
``U (z_i - z_j)``, filtered by the crossed CTF and standing on the noise floor the
measurement carries (see :mod:`sinovar.refine.kernel`). Fitting the ``z`` so that
those modelled distances match the measured ones is what this module does.

Two things follow from modelling rather than idealising the measurement. The fit
is two-sided --- a common line is blind to most of a discrepancy and the CTF
suppresses part of what is left, but both are now *in the model*, so a
measurement below the model is as much of an error as one above it. And the
metric is not Euclidean: each pair sees the basis only through the line it cuts,
so it constrains ``z_i - z_j`` through a ``(K, K)`` positive semi-definite matrix
rather than through a length. The optimizer is therefore SMACOF in structure ---
one exact per-point majorization solve, no step size to tune --- but Mahalanobis
in its pair term.

Sampling
--------
The full problem is ``N^2 / 2`` pairs, of which a 100k dataset has 5e9, and each
pair needs a common line of the basis. Nothing like that is affordable, and
nothing like that is necessary: ``K`` coordinates per particle need ``O(K)``
well-conditioned pairs to be determined, not ``N``. Each step therefore draws a
few *anchors* and, for each, a few hundred random *partners*, and updates the
anchors alone --- exactly the row-batch structure of
:mod:`sinovar.embedding.smacof`, with the row subsampled as well. An epoch is
``N / anchors`` steps, so every particle is updated once per epoch.

The anchor is what makes the sampling asymmetric, and it is the projections that
ask for it: a particle's common lines with *every* other particle lie in its own
image plane, so one ``box^3`` projection of the whole basis serves all of an
anchor's partners and the per-pair cost falls to a ``box^2`` line integral. The
work per step is ``anchors * (K * box^3 + partners * K * box^2)``, which is why
partners are counted in hundreds: that is where the two terms balance.

Because the measured distances are needed for a vanishing fraction of the pairs,
they are computed as the pairs are drawn rather than read back from a
precomputed matrix. That also keeps the measurement exactly the crossed-CTF
comparison this model inverts, rather than the Mahalanobis variant the
``distance`` command writes.

Scale
-----
The images carry an arbitrary scale, the eigen-volumes another, so a single
global factor ``s`` relates a modelled distance to a measured one. It is fitted
once, by least squares on the squared distances of the first few thousand sampled
pairs, and then held fixed --- which is what pins the scale of the refined
``z``. Holding it fixed is the point: with ``s`` free, ``(s / a, a z)`` would fit
identically for any ``a``, and the refined coordinates would drift out of the
units in which ``U z`` reconstructs a volume.

That degeneracy is also the limit of what a fitted scale can do: since it is
fitted *from the input embedding*, a global scale error in that input is absorbed
into ``s`` rather than corrected. The refinement fixes the shape of an embedding
in the units it arrived in; pass ``scale`` explicitly to fix the units too.

For the same reason the components are never rotated or reordered on the way out,
unlike the embeddings of :mod:`sinovar.embedding`: component ``k`` means
eigen-volume ``k`` and nothing else. Only the translation the objective cannot
see is fixed, by restoring the input's centroid.
"""
from typing import Dict, Iterator, Optional, Tuple
import logging

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ..filter import rfft_multiplicity
from .kernel import refine_step

logger = logging.getLogger(__name__)


def _iter_pair_batches(
    n: int,
    anchors_per_step: int,
    partners_per_anchor: int,
    n_steps: int,
    rng: np.random.Generator,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``n_steps`` batches of ``(anchors, partners)`` indices.

    Anchors are consumed from a permutation, redrawn once it no longer holds a
    full batch, so an epoch updates every particle once; they are sorted, which
    keeps their image gather in file order. Partners are drawn uniformly and
    independently --- an anchor's own index is stepped over, but a partner
    repeated within a batch is harmless, it simply counts twice.
    """
    order = rng.permutation(n)
    cursor = 0
    for _ in range(n_steps):
        if cursor + anchors_per_step > order.shape[0]:
            order = rng.permutation(n)
            cursor = 0

        anchors = np.sort(order[cursor:cursor + anchors_per_step])
        cursor += anchors_per_step

        partners = rng.integers(
            0, n, size=(anchors.shape[0], partners_per_anchor)
        )
        collision = partners == anchors[:, None]
        partners = np.where(collision, (partners + 1) % n, partners)
        yield anchors, partners


def _accumulate(
    totals: Dict[str, jax.Array],
    stats: Dict[str, jax.Array]
) -> Dict[str, jax.Array]:
    """Sum a step's statistics into ``totals``, on the device.

    Keeping them there is what lets the loop run ahead: reading a statistic back
    would block on the step that produced it, and with it the host-side gather of
    the next batch of images.
    """
    return {
        key: value if key not in totals else totals[key] + value
        for key, value in stats.items()
    }


def refine_embedding(
    embedding: np.ndarray,
    volumes: np.ndarray,
    images: np.ndarray,
    rotations: np.ndarray,
    shifts: np.ndarray,
    ctf: np.ndarray,
    noise_variance: np.ndarray,
    mask: Optional[np.ndarray] = None,
    frequency_weights: Optional[np.ndarray] = None,
    n_epochs: int = 4,
    anchors_per_step: int = 8,
    partners_per_anchor: int = 256,
    ridge: float = 1e-2,
    huber: float = 1.0,
    step_size: float = 1.0,
    scale: Optional[float] = None,
    noise_floor: bool = True,
    n_scale_steps: int = 8,
    partner_batch: int = 8,
    anchor_batch: int = 1,
    slab: int = 8,
    random_state: int = 0,
    progress: bool = True,
) -> np.ndarray:
    """Refine an embedding so its modelled common-line distances match the data.

    Every array must already be on a common working grid of box size ``B``: the
    images and the eigen-volumes resampled to it (see
    :mod:`sinovar.resample`), the shifts in its pixels, and the CTF, the noise
    variance and the weights on its ``F = B // 2 + 1`` frequencies.

    Args:
        embedding: initial ``(N, K)`` coordinates, e.g. RECOVAR's ``z``.
        volumes: ``(K, B, B, B)`` eigen-volume basis, in the same order as the
            embedding's components.
        images: ``(N, B, B)`` particle images. Read in batches and never
            uploaded whole, so this may stay in host memory.
        rotations: ``(N, 3, 3)`` orientation matrices.
        shifts: ``(N, 2)`` origin shifts in working-box pixels.
        ctf: ``(N, F)`` 1D CTFs.
        noise_variance: ``(F,)`` per-frequency variance of an experimental line,
            which sets each pair's noise floor.
        mask: optional ``(B, B)`` circular mask, applied exactly as the distance
            kernel applies it; must match the one ``noise_variance`` was
            calibrated with.
        frequency_weights: optional ``(F,)`` weights, e.g. a squared low-pass.
        n_epochs: passes over the dataset. Each updates every particle once, so
            this is the iteration count *and* the cost: ``n_epochs * N`` anchor
            projections of the basis.
        anchors_per_step: particles updated per step.
        partners_per_anchor: pairs drawn for each of them, which is how much
            information one update has. Below ``K`` the solve leans on ``ridge``.
        ridge: damping of the per-anchor solve, relative to the mean curvature
            of its pairs.
        huber: residual at which a pair's pull saturates, in units of the RMS
            measured distance.
        step_size: relaxation of the majorization step; 1 minimises the
            majorizer exactly, below that damps it.
        scale: global factor from volume units to image units. Fitted from the
            data when ``None``, which is normally what you want --- at the price
            of absorbing any global scale error of the input embedding, since
            the two are indistinguishable to the objective.
        noise_floor: model the noise the measurement carries. Turning it off
            fits the raw measurements, which biases the embedding outwards by
            whatever the noise contributes.
        n_scale_steps: steps sampled to fit ``scale`` before refining. Their
            coordinate updates are discarded, since the scale they were computed
            with is still wrong.
        partner_batch, anchor_batch, slab: device-memory knobs; see
            :func:`sinovar.refine.kernel.refine_step`.
        random_state: seed for the anchor and partner sampling.
        progress: show a progress bar.

    Returns:
        The refined ``(N, K)`` embedding, in the same component order and with
        the same centroid as the input.
    """
    n, n_components = embedding.shape
    box_size = images.shape[-1]
    n_frequencies = box_size // 2 + 1

    if volumes.shape != (n_components, box_size, box_size, box_size):
        raise ValueError(
            f'volumes must be {(n_components, box_size, box_size, box_size)} to '
            f'match a {n_components}-component embedding of box {box_size}, got '
            f'{tuple(volumes.shape)}'
        )
    if images.shape != (n, box_size, box_size):
        raise ValueError(f'images must be {(n, box_size, box_size)}')
    if rotations.shape != (n, 3, 3) or shifts.shape != (n, 2):
        raise ValueError('rotations must be (N, 3, 3) and shifts (N, 2)')
    if ctf.shape != (n, n_frequencies):
        raise ValueError(f'ctf must be {(n, n_frequencies)}')
    if noise_variance.shape != (n_frequencies,):
        raise ValueError(f'noise_variance must be {(n_frequencies,)}')
    if mask is not None and mask.shape != (box_size, box_size):
        raise ValueError(f'mask must be {(box_size, box_size)}')
    if n_epochs < 1:
        raise ValueError('n_epochs must be >= 1')
    if anchors_per_step < 1 or partners_per_anchor < 1:
        raise ValueError('anchors_per_step and partners_per_anchor must be >= 1')
    if n < 2:
        raise ValueError('need at least 2 particles to refine')
    if ridge <= 0.0:
        raise ValueError('ridge must be > 0; the per-anchor solve is damped by it')
    if huber <= 0.0:
        raise ValueError('huber must be > 0')
    if not 0.0 < step_size <= 2.0:
        raise ValueError('step_size must be in (0, 2]')
    if scale is not None and not scale > 0.0:
        raise ValueError('scale must be > 0')

    anchors_per_step = min(anchors_per_step, n)
    steps_per_epoch = max(1, n // anchors_per_step)
    n_steps = n_epochs * steps_per_epoch

    weight = rfft_multiplicity(box_size)
    if frequency_weights is not None:
        weight = weight * jnp.asarray(frequency_weights)

    # Everything but the images is small enough to live on the device for the
    # whole run; the images are gathered per step and uploaded with the batch.
    coords = jnp.asarray(embedding, dtype=jnp.float32)
    device = coords.device
    volumes = jax.device_put(jnp.asarray(volumes, dtype=jnp.float32), device)
    rotations = jax.device_put(jnp.asarray(rotations, dtype=jnp.float32), device)
    shifts = jax.device_put(jnp.asarray(shifts, dtype=jnp.float32), device)
    ctf = jax.device_put(jnp.asarray(ctf, dtype=jnp.float32), device)
    weight = jax.device_put(jnp.asarray(weight, dtype=jnp.float32), device)
    mask = None if mask is None else jax.device_put(
        jnp.asarray(mask, dtype=jnp.float32), device
    )
    variance = jnp.asarray(noise_variance, dtype=jnp.float32)
    if not noise_floor:
        variance = jnp.zeros_like(variance)
    variance = jax.device_put(variance, device)

    logger.info(
        'Refining %d particle(s) of %d component(s) over %d epoch(s) of %d '
        'step(s): %.3g pair(s) and %.3g basis projection(s) on %s',
        n, n_components, n_epochs, steps_per_epoch,
        n_steps * anchors_per_step * partners_per_anchor,
        n_steps * anchors_per_step, device.device_kind,
    )

    rng = np.random.default_rng(random_state)
    scale2 = jnp.float32(1.0 if scale is None else scale * scale)
    # Both are set from the measurements themselves, once the scale fit has seen
    # them; until then nothing caps a pull, and the floor only guards 0/0.
    kappa = jnp.float32(jnp.inf)
    distance_floor = jnp.float32(jnp.finfo(jnp.float32).tiny)
    ridge_ = jnp.float32(ridge)
    step_size_ = jnp.float32(step_size)

    def run_step(anchors, partners):
        return refine_step(
            coords,
            volumes,
            jax.device_put(jnp.asarray(anchors), device),
            jax.device_put(jnp.asarray(partners), device),
            jax.device_put(jnp.asarray(images[anchors], dtype=jnp.float32), device),
            jax.device_put(jnp.asarray(images[partners], dtype=jnp.float32), device),
            rotations,
            shifts,
            ctf,
            weight,
            variance,
            scale2,
            kappa,
            ridge_,
            step_size_,
            distance_floor,
            mask=mask,
            partner_batch=min(partner_batch, partners_per_anchor),
            anchor_batch=min(anchor_batch, anchors_per_step),
            slab=slab,
        )

    # --- scale fit: the same pair machinery, with the updates thrown away.
    logger.info('Fitting the global scale over %d step(s)', n_scale_steps)
    totals: Dict[str, jax.Array] = {}
    for anchors, partners in _iter_pair_batches(
        n, anchors_per_step, partners_per_anchor, n_scale_steps, rng
    ):
        _, stats = run_step(anchors, partners)
        totals = _accumulate(totals, stats)

    n_pairs = n_scale_steps * anchors_per_step * partners_per_anchor
    rms_measured = float(jnp.sqrt(jnp.maximum(totals['measured2'] / n_pairs, 0.0)))
    if not rms_measured > 0.0:
        raise ValueError('every measured common-line distance is zero')

    if scale is None:
        correction = float(totals['fit_numerator']) / float(totals['fit_denominator'])
        if correction > 0.0:
            scale2 = jnp.float32(float(scale2) * correction)
        else:
            logger.warning(
                'The modelled distances do not correlate with the measured ones '
                '(scale correction %.3g); keeping the scale at 1. Check that the '
                'eigen-volumes match the embedding and its component order.',
                correction,
            )
    logger.info(
        'Global scale %.4g (volume units to image units), RMS measured distance '
        '%.4g', float(jnp.sqrt(scale2)), rms_measured,
    )

    kappa = jnp.float32(huber * rms_measured)
    distance_floor = jnp.float32(1e-6 * rms_measured)

    # --- refinement
    log_every = max(1, n_steps // 20)
    bar = tqdm(
        total=n_steps, disable=not progress, unit='step', desc='Refinement steps'
    )
    totals = {}
    n_pairs = 0
    n_anchors = 0
    with bar:
        for step, (anchors, partners) in enumerate(_iter_pair_batches(
            n, anchors_per_step, partners_per_anchor, n_steps, rng
        )):
            updated, stats = run_step(anchors, partners)
            coords = coords.at[jnp.asarray(anchors)].set(updated)

            totals = _accumulate(totals, stats)
            n_pairs += anchors.shape[0] * partners_per_anchor
            n_anchors += anchors.shape[0]
            bar.update(1)

            if step % log_every == 0 or step == n_steps - 1:
                radius = float(jnp.sqrt(jnp.mean(jnp.sum(
                    jnp.square(coords), axis=-1
                ))))
                logger.info(
                    'epoch %5.1f/%d: residual %.4g of RMS distance %.4g '
                    '(model %.4g), step %.3g of radius %.4g',
                    (step + 1) / steps_per_epoch, n_epochs,
                    float(jnp.sqrt(totals['stress'] / n_pairs)),
                    float(jnp.sqrt(totals['measured2'] / n_pairs)),
                    float(jnp.sqrt(totals['model2'] / n_pairs)),
                    float(jnp.sqrt(totals['increment2'] / n_anchors)), radius,
                )
                totals = {}
                n_pairs = 0
                n_anchors = 0

    # The objective only sees differences, so the translation is free; keep the
    # input's centroid rather than letting it drift. The axes are *not* touched:
    # component k is eigen-volume k.
    refined = np.asarray(coords, dtype=np.float64)
    refined += embedding.mean(axis=0) - refined.mean(axis=0)

    moved = np.sqrt(np.mean(np.sum(np.square(refined - embedding), axis=-1)))
    radius = np.sqrt(np.mean(np.sum(np.square(embedding), axis=-1)))
    logger.info(
        'Refinement moved the embedding by %.4g (%.1f%% of its RMS radius); '
        'component spread %s -> %s',
        moved,
        100.0 * moved / radius,
        np.array2string(embedding.std(axis=0), precision=3),
        np.array2string(refined.std(axis=0), precision=3),
    )
    return refined
