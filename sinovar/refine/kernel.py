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

    (sum_j M_ij + lambda I) (x_i - z_i) = sum_j pull_ij M_ij (z_i - z_j),
    pull_ij = clip(delta_ij - d_ij, -kappa, kappa) / d_ij,

i.e. a Gauss-Newton step that needs no line search: with the measurements
matched exactly every ``pull`` vanishes and the anchor stays put. Solving for the
*increment* rather than for ``x_i`` is what keeps that exact --- the two forms
differ by ``sum_j M_ij z_i``, a large quantity that would have to cancel.

``kappa`` caps how far one pair can pull, which matters here because a common
line is a noisy measurement whose upward errors are not symmetric in effect, and
``lambda = ridge * trace(sum_j M_ij) / K`` damps the step along the directions
the sampled pairs constrain weakly --- with ``K`` components and one ``K``-vector
of information per pair, the sum is close to singular until enough partners have
been drawn.
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

#: Layout of the per-pair statistics accumulated inside a step.
_STATISTICS = (
    'measured2',        # sum delta^2, the measured squared distances
    'model2',           # sum d^2, the modelled squared distances
    'stress',           # sum (delta - d)^2, the objective
    'fit_denominator',  # sum q^2         with q = d^2 - c, the modelled signal
    'fit_numerator',    # sum q (delta^2 - c), for the global scale fit
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

            measured = jnp.sqrt(jnp.maximum(measured2, 0.0))
            model = jnp.sqrt(jnp.maximum(signal2 + floor, 0.0))
            residual = measured - model
            # Two-sided: the model is not a bound, so it is pulled out when the
            # measurement is larger and in when it is smaller, and no further
            # than kappa either way.
            pull = (
                jnp.clip(residual, -kappa, kappa)
                / jnp.maximum(model, distance_floor)
            )

            statistics = jnp.stack((
                measured2,
                signal2 + floor,
                jnp.square(residual),
                jnp.square(signal2),
                signal2 * (measured2 - floor),
            ))
            return metric, pull * metric_offset, statistics

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
