"""One majorization step of the common-line refinement.

Everything a step needs beyond the images it was handed is device resident --- the
basis, the poses, the CTFs, the coordinates --- so a step is a single dispatch
that touches ``anchors * partners`` pairs and returns the anchors' new
coordinates.

The pair term
-------------
A pair ``(i, j)`` is compared along its common line only. The measurement is the
ManifoldEM-style crossed-CTF difference of the two experimental lines,

    delta_ij^2 = sum_f w_f |C_j Y_i - C_i Y_j|^2,

which is what :mod:`sinovar.distance` computes bar its Mahalanobis denominator.
Substituting the image formation ``Y = C * l(V)`` shows what the same quantity is
in terms of the volumes: the crossed product cancels the individual CTFs and
leaves a *doubly* filtered difference of the two volumes' common lines,

    C_j Y_i - C_i Y_j = C_i C_j (l_ij V_i - l_ij V_j) = C_i C_j l_ij U (z_i - z_j),

so the model of the measurement is

    d_ij^2(z) = sum_f w_f s^2 C_i^2 C_j^2 |A_ij (z_i - z_j)|_f^2 + c_ij,
    A_ij = rfft(l_ij U),
    c_ij = sum_f w_f (C_i^2 + C_j^2) sigma2_f.

``A_ij`` is the basis' own common line, extracted from the anchor's projections
by :mod:`sinovar.volume.project`; ``s`` converts the arbitrary scale of the
images into that of the volumes; and ``c_ij`` is the noise the measurement
carries, which is *known* (the same ``sigma2`` the distance command calibrates)
and therefore modelled rather than subtracted --- a measurement below the floor
then costs nothing to explain, where subtracting would have had to clip it.

Both terms carry a scalar this cannot know a priori: ``s`` because the images
have an arbitrary normalization, and ``sigma2`` because it is estimated outside
the particle mask, where the background is not quite what it is underneath.
Both are calibrated together, by least squares over sampled pairs against

    delta_ij^2 ~ s^2 q_ij + beta c_ij,    q_ij = |A_ij (z_i - z_j)|^2_w,

which is possible because the two terms depend differently on the pair: ``q``
carries the geometry and ``C_i^2 C_j^2``, ``c`` only ``C_i^2 + C_j^2``. What
separates them is therefore already there in a dataset of a single defocus, where
``c`` is constant across pairs and ``q`` is not --- a slope and an offset --- but
a spread of defocus varies ``c`` too and is what makes ``beta`` well determined.
Only their sum is identifiable in the limit where the two become proportional,
which :mod:`sinovar.refine.refine` detects and reports rather than fitting
through.

``beta`` matters out of proportion to its size whenever the measurement is
noise-dominated: ``s^2 q = delta^2 - beta c`` is then a difference of large
nearly-equal numbers, so a few percent of error in ``sigma2`` is a large
fractional error in every modelled distance --- and, being one-signed, it does
not average out but inflates the embedding.

Writing the first term as a quadratic form ``(z_i - z_j)' M_ij (z_i - z_j)``
with ``M_ij = Re(A_ij^H W A_ij)`` a ``(K, K)`` positive semi-definite matrix
turns the model into a per-pair *Mahalanobis* distance: the common line sees
only the ``K``-dimensional shape of the basis it happens to cut, and is blind to
the rest. That is the whole point of refining against it --- and the reason the
model is no longer a lower bound of anything, so the fit is two-sided.

The update
----------
For the stress ``sum (delta - d(X))^2`` the Guttman majorization needs nothing
more than that ``d`` be a seminorm of ``z_i - z_j`` plus a constant: Cauchy-
Schwarz bounds ``d(X)`` below by its linearization at the current iterate, and
the constant ``c_ij`` drops out of the gradient because it lives in a direction
``M_ij`` cannot see. Minimising the majorizer in one anchor's coordinates with
every other point held fixed is then a ``(K, K)`` linear solve,

    (sum_j g_ij M_ij + lambda I) (x_i - z_i)
        = sum_j g_ij pull_ij M_ij (z_i - z_j),
    pull_ij = clip(delta_ij - d_ij, -kappa, kappa) / d_ij,

i.e. a Gauss-Newton step that needs no line search: with the measurements
matched exactly every ``pull`` vanishes and the anchor stays put. Solving for the
*increment* rather than for ``x_i`` is what keeps that exact --- the two forms
differ by ``sum_j M_ij z_i``, a large quantity that would have to cancel.

``kappa`` caps how far one pair can pull, which matters here because a common
line is a noisy measurement whose upward errors are not symmetric in effect, and
``lambda = ridge * trace(sum_j g_ij M_ij) / K`` damps the step along the
directions the sampled pairs constrain weakly --- with ``K`` components and one
``K``-vector of information per pair, the sum is close to singular until enough
partners have been drawn.

Weighting the pairs
-------------------
``g_ij`` is what keeps a noise-dominated dataset from diluting its own updates.
A pair whose modelled signal sits below its noise floor still carries a full
``M_ij`` and a ``pull`` that is almost entirely measurement scatter --- of
relative size ``1 / (2 sqrt(F_eff))`` with ``F_eff`` the effective number of
frequency samples in a common line, some tens --- so at uniform weight the
informative pairs are averaged against pairs that only add variance. Trusting
each pair in proportion to how much of its modelled distance is signal,

    g_ij = (q_ij + floor_weight c_ij) / (q_ij + c_ij),   q_ij = d_ij^2 - c_ij,

concentrates an update on the pairs that carry information. It costs nothing:
both quantities are already computed for the model itself.

Two choices in that expression are deliberate. The weight is built from the
*modelled* signal rather than from ``delta^2 - c``, because a weight read off
the measurement would silence whichever pairs happened to fluctuate downwards
and keep those that fluctuated up --- the rectification that
:mod:`sinovar.embedding.smacof` documents at length, reappearing as a
selection effect. And ``floor_weight`` keeps a share of every pair alive, since
a weight strictly proportional to the modelled signal lets a badly placed point
suppress the very pairs that would pull it back, and because ``q_ij`` is small
both when the pair carries no signal *and* when the current offset happens to
lie in the directions its common line cannot see. At ``floor_weight = 1`` the
weighting is off, and it switches itself off anyway wherever the floor vanishes.
"""
from functools import partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from ..geometry.common_lines import compute_intrinsic_common_line_angles
from ..sinogram.project import _project_image
from ..volume.project import project_volume_basis

# Every volume of the basis, one shared common-line angle and no shift: the
# basis projections are centred by construction.
_project_basis_lines = jax.vmap(_project_image, in_axes=(0, None, None, None))

#: Layout of the per-pair statistics accumulated inside a step. The last five are
#: the normal equations of the calibration fit ``delta^2 ~ s2 q + beta c``, with
#: ``q`` the modelled signal at unit scale and ``c`` the noise floor.
_STATISTICS = (
    'measured2',        # sum delta^2, the measured squared distances
    'model2',           # sum d^2, the modelled squared distances
    'stress',           # sum (delta - d)^2, the objective
    'weight',           # sum g, how much of the sampling carried information
    'floor',            # sum c, how much of the measurement the noise explains
    'qq',               # sum q^2
    'qc',               # sum q c
    'cc',               # sum c^2
    'q_measured2',      # sum q delta^2
    'c_measured2',      # sum c delta^2
)


@partial(jax.jit, static_argnames=('partner_batch', 'anchor_batch', 'slab'))
def refine_step(
    coords: jax.Array,
    volumes: jax.Array,
    anchors: jax.Array,
    partners: jax.Array,
    images_anchor: jax.Array,
    images_partner: jax.Array,
    rotations: jax.Array,
    shifts: jax.Array,
    ctf: jax.Array,
    weight: jax.Array,
    noise_variance: jax.Array,
    scale2: jax.Array,
    kappa: jax.Array,
    ridge: jax.Array,
    step_size: jax.Array,
    distance_floor: jax.Array,
    floor_weight: jax.Array,
    mask: Optional[jax.Array] = None,
    partner_batch: int = 8,
    anchor_batch: int = 1,
    slab: int = 8,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Update the anchors' embedding coordinates from their sampled pairs.

    Parameters
    ----------
    coords:
        Current embedding, ``(N, K)``; only the anchors' rows are read back.
    volumes:
        Basis of eigen-volumes, ``(K, B, B, B)``.
    anchors:
        Indices of the particles updated by this step, ``(b,)``.
    partners:
        Indices of the particles each anchor is compared against, ``(b, J)``.
    images_anchor, images_partner:
        The corresponding images, ``(b, B, B)`` and ``(b, J, B, B)``. Passed in
        rather than indexed out of a device-resident stack so that the host can
        hold the dataset.
    rotations, shifts, ctf:
        Per-particle pose and 1D CTF for the *whole* dataset, ``(N, 3, 3)``,
        ``(N, 2)`` and ``(N, F)``; small enough to stay on the device.
    weight:
        Per-frequency weight of the line comparison, ``(F,)``: the rfft
        multiplicity times any low-pass filter (squared, since it applies to
        both lines of the difference).
    noise_variance:
        Per-frequency variance ``sigma2`` of an experimental line, ``(F,)``,
        which sets the noise floor each pair's model carries. Zero to fit the
        raw measurements instead.
    scale2:
        Squared global scale converting volume units into image units.
    kappa:
        Residual, as a distance, at which a pair's pull saturates.
    ridge:
        Damping of the solve, relative to the mean curvature of the anchor's
        pairs.
    step_size:
        Relaxation of the increment; 1 minimises the majorizer exactly.
    distance_floor:
        Lower bound on the modelled distance dividing a pull, guarding 0/0 for
        coincident points.
    floor_weight:
        Share of its weight a pair keeps when its modelled signal vanishes into
        the noise floor. One weights every pair alike.
    mask:
        Circular real-space mask, ``(B, B)``, applied in the centred projection
        frame exactly as the distance kernel applies it.
    partner_batch, anchor_batch:
        Partners and anchors processed together. Peak memory grows with both;
        the result does not depend on either.
    slab:
        Integration steps resampled at once when projecting the basis.

    Returns
    -------
    tuple
        The anchors' new coordinates, ``(b, K)``, and the statistics summed over
        every pair of the step.
    """
    n_components = coords.shape[-1]
    identity = jnp.eye(n_components, dtype=coords.dtype)
    zero_shift = jnp.zeros_like(shifts[0])

    def one_anchor(anchor):
        index, image_anchor, partner_index, partner_images = anchor

        rotation = rotations[index]
        shift = shifts[index]
        ctf_anchor = ctf[index]
        ctf2_anchor = jnp.square(ctf_anchor)
        z_anchor = coords[index]

        # The one ``box^3`` operation of the pair, shared by every partner: all
        # of an anchor's common lines lie in its own image plane.
        projections = project_volume_basis(volumes, rotation, slab=slab)

        angle_anchor, angle_partner, _ = compute_intrinsic_common_line_angles(
            rotation[None],               # (1, 3, 3)
            rotations[partner_index],     # (J, 3, 3)
        )                                 # each (J,)

        def one_partner(partner):
            other, image_partner, angle_here, angle_there = partner
            ctf_partner = ctf[other]
            ctf2_partner = jnp.square(ctf_partner)

            # Measurement: the crossed-CTF difference of the two lines, which is
            # blind to the individual CTFs and to the reference the images lack.
            line_anchor = _project_image(image_anchor, shift, angle_here, mask)
            line_partner = _project_image(
                image_partner, shifts[other], angle_there, mask
            )
            difference = (
                ctf_partner * jnp.fft.rfft(line_anchor)
                - ctf_anchor * jnp.fft.rfft(line_partner)
            )
            measured2 = jnp.sum(
                weight * (
                    jnp.square(difference.real) + jnp.square(difference.imag)
                )
            )

            # Model: the basis' own common line, carrying the crossed CTF the
            # measurement built in and the noise floor it cannot see past.
            lines = _project_basis_lines(
                projections, zero_shift, angle_here, mask
            )                                                    # (K, B)
            basis = jnp.fft.rfft(lines, axis=-1)                 # (K, F)
            gain = jnp.sqrt(scale2 * weight * ctf2_anchor * ctf2_partner)
            weighted = basis * gain                              # (K, F)
            metric = (
                weighted.real @ weighted.real.T
                + weighted.imag @ weighted.imag.T
            )                                                    # (K, K)
            floor = jnp.sum(weight * (ctf2_anchor + ctf2_partner) * noise_variance)

            offset = z_anchor - coords[other]
            metric_offset = metric @ offset
            signal2 = jnp.dot(offset, metric_offset)

            model2 = signal2 + floor
            measured = jnp.sqrt(jnp.maximum(measured2, 0.0))
            model = jnp.sqrt(jnp.maximum(model2, 0.0))
            residual = measured - model
            # Two-sided: the model is not a bound, so it is pulled out when the
            # measurement is larger and in when it is smaller, and no further
            # than kappa either way.
            pull = (
                jnp.clip(residual, -kappa, kappa)
                / jnp.maximum(model, distance_floor)
            )

            # How much of this pair's modelled distance is signal rather than
            # noise, which is how far its residual is worth trusting. Neutral
            # where there is no floor to speak of --- including the degenerate
            # pair that models nothing at all.
            pair_weight = jnp.where(
                model2 > 0.0,
                (signal2 + floor_weight * floor)
                / jnp.maximum(model2, jnp.finfo(coords.dtype).tiny),
                1.0,
            )

            statistics = jnp.stack((
                measured2,
                model2,
                jnp.square(residual),
                pair_weight,
                floor,
                jnp.square(signal2),
                signal2 * floor,
                jnp.square(floor),
                signal2 * measured2,
                floor * measured2,
            ))
            # The statistics stay unweighted: the calibration they feed is a
            # regression over pairs in its own right, and its regressors are
            # exactly what the weight is built out of.
            return (
                pair_weight * metric,
                pair_weight * pull * metric_offset,
                statistics,
            )

        metric, force, statistics = jax.lax.map(
            one_partner,
            (partner_index, partner_images, angle_anchor, angle_partner),
            batch_size=partner_batch,
        )

        # Curvature and force of the anchor's own coordinates; symmetrised so
        # that the solve can rely on the positive-definiteness the sum has by
        # construction.
        curvature = jnp.sum(metric, axis=0)
        curvature = 0.5 * (curvature + curvature.T)
        damping = (
            ridge * jnp.trace(curvature) / n_components
            + jnp.finfo(coords.dtype).tiny
        )
        increment = jax.scipy.linalg.solve(
            curvature + damping * identity,
            jnp.sum(force, axis=0),
            assume_a='pos',
        )
        return (
            z_anchor + step_size * increment,
            (jnp.sum(statistics, axis=0), jnp.sum(jnp.square(increment))),
        )

    updated, (statistics, increment2) = jax.lax.map(
        one_anchor,
        (anchors, images_anchor, partners, images_partner),
        batch_size=anchor_batch,
    )

    totals = jnp.sum(statistics, axis=0)
    stats = dict(zip(_STATISTICS, totals))
    stats['increment2'] = jnp.sum(increment2)
    return updated, stats
