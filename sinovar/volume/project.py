"""Real-space projection of a volume along a particle's viewing direction.

A common line of the model is a *double* projection: the volume is projected
onto the image plane of one of the two particles, and that projection is then
projected onto the common-line axis by
:func:`sinovar.sinogram.project._project_image`. Since the common-line axis of a
pair lies in *both* image planes, the second step can be taken in either
particle's frame and the two agree --- which is exactly the self-consistency
this module is tested against.

Doing it in two steps rather than one is what makes the model affordable. The
line only depends on the common-line direction, and all the directions a
particle shares with any other lie in its own image plane, so a *single* volume
projection per particle serves every pair it takes part in: the ``box^3`` work
is paid once and the per-pair cost drops to a ``box^2`` line integral. That is
also why the projection is done in real space --- it is the same operator, with
the same bilinear interpolation and the same mask, that the experimental lines
go through, so the model matches the measurement rather than an idealization of
it.

The rotation convention is the one the rest of the codebase uses (and which
:mod:`sinovar.geometry.common_lines` assumes): the *rows* of the matrix are the
image frame's axes expressed in volume coordinates, i.e. row 0 is the image's
``x`` (column) axis, row 1 its ``y`` (row) axis and row 2 the viewing direction
integrated over. The identity therefore projects a ``(z, y, x)`` volume by
summing over ``z``.
"""
from functools import partial

import jax
import jax.numpy as jnp

# One set of sample positions, every volume of the basis: the coordinates are
# built once and shared, so a basis costs one gather per volume and no extra
# arithmetic.
_gather_basis = jax.vmap(
    lambda volume, coordinates: jax.scipy.ndimage.map_coordinates(
        volume, coordinates, order=1, mode='constant', cval=0.0
    ),
    in_axes=(0, None),
)


@partial(jax.jit, static_argnames=('slab',))
def project_volume_basis(
    volumes: jax.Array,
    rotation: jax.Array,
    slab: int = 8,
) -> jax.Array:
    """Project every volume of a basis along one orientation.

    Parameters
    ----------
    volumes:
        Cubic volumes, ``(K, B, B, B)``, indexed ``(z, y, x)``.
    rotation:
        Orientation matrix, ``(3, 3)``, whose rows are the image axes in volume
        coordinates.
    slab:
        Number of integration steps resampled at once. Peak memory is
        ``K * slab * B^2``; the total work does not depend on it.

    Returns
    -------
    jax.Array
        Projections, ``(K, B, B)``, indexed ``(y, x)`` like a particle image.
    """
    n_volumes, box = volumes.shape[0], volumes.shape[-1]
    if volumes.ndim != 4 or any(d != box for d in volumes.shape[1:]):
        raise ValueError(f'volumes must be (K, B, B, B), got {volumes.shape}')

    centre = 0.5 * (box - 1)
    grid = jnp.arange(box, dtype=volumes.dtype) - centre

    # Sample positions of one image plane, in volume coordinates: pixel (y, x)
    # sits at x * row0 + y * row1, and the integration walks along row2.
    in_plane = (
        grid[None, :, None] * rotation[0]     # (1, B, 3) along the image x axis
        + grid[:, None, None] * rotation[1]   # (B, 1, 3) along the image y axis
    )                                         # (B, B, 3)

    def one_slab(total, offsets):
        position = in_plane + offsets[:, None, None, None] * rotation[2]
        # map_coordinates indexes the array, so the Cartesian components are
        # reordered into the volume's (z, y, x) axis order.
        coordinates = jnp.stack(
            (
                position[..., 2] + centre,
                position[..., 1] + centre,
                position[..., 0] + centre,
            )
        )                                             # (3, slab, B, B)
        samples = _gather_basis(volumes, coordinates)  # (K, slab, B, B)
        return total + jnp.sum(samples, axis=1), None

    # Padding the integration with positions outside the volume keeps the scan
    # rectangular; ``mode='constant'`` makes them contribute nothing.
    n_slabs = -(-box // slab)  # ceil division
    offsets = jnp.concatenate(
        (grid, jnp.full((n_slabs * slab - box,), 2.0 * box, grid.dtype))
    ).reshape(n_slabs, slab)

    total = jnp.zeros((n_volumes, box, box), volumes.dtype)
    projections, _ = jax.lax.scan(one_slab, total, offsets)
    return projections


@partial(jax.jit, static_argnames=('slab',))
def project_volume(
    volume: jax.Array,
    rotation: jax.Array,
    slab: int = 8,
) -> jax.Array:
    """Project a single ``(B, B, B)`` volume; see :func:`project_volume_basis`."""
    return project_volume_basis(volume[None], rotation, slab=slab)[0]
