from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp

from ..filter import rfft_multiplicity
from ..sinogram.project import _project_image
from ..geometry.common_lines import compute_intrinsic_common_line_angles

# One angle per image: (N, H, W), (N, 2), (N,), (H, W) -> (N, box). Each image
# is projected along its own single common-line angle.
_project_lines = jax.vmap(_project_image, in_axes=(0, 0, 0, None))
# One image, a fan of angles: (H, W), (2,), (M,), (H, W) -> (M, box).
_project_fan = jax.vmap(_project_image, in_axes=(None, None, 0, None))


@partial(jax.jit, static_argnames=('col_batch',))
def compute_distance2_tile(
    images_row: jax.Array,
    shifts_row: jax.Array,
    rotations_row: jax.Array,
    ctf_row: jax.Array,
    images_col: jax.Array,
    shifts_col: jax.Array,
    rotations_col: jax.Array,
    ctf_col: jax.Array,
    sigma2: jax.Array,
    mask: Optional[jax.Array] = None,
    frequency_weights: Optional[jax.Array] = None,
    col_batch: int = 8,
) -> jax.Array:
    """CTF-weighted common-line distance for every (row, col) pair in a tile.

    Returns a ``(n_row, n_col)`` block of squared distances ``D[i, j]`` between
    row particle ``i`` and column particle ``j``.

    The common-line orientation depends on *both* members of a pair, so every
    cell needs its own pair of 1D projections; nothing is shared between cells
    beyond the per-particle CTF. Materialising all ``n_row * n_col`` resampled
    ``(H, W)`` projection frames at once costs ``n_row * n_col * H * W`` and is
    what limits the block size. Instead the tile is swept one group of columns
    at a time: :func:`jax.lax.map` vectorises ``col_batch`` columns and scans
    over the groups, so peak projection memory is only
    ``col_batch * n_row * H * W`` and grows with ``col_batch`` rather than with
    the block size. ``col_batch`` trades that peak against parallelism.

    Parameters
    ----------
    images_row, images_col:
        Real-space particle images, ``(n_row, H, W)`` / ``(n_col, H, W)``.
    shifts_row, shifts_col:
        Per-particle origin shifts in pixels, ``(n, 2)``.
    rotations_row, rotations_col:
        Per-particle orientation matrices, ``(n, 3, 3)``.
    ctf_row, ctf_col:
        Pre-computed 1D CTFs, ``(n, F)`` with ``F`` the rfft length of a
        projection. Passed in so they are computed once per particle.
    sigma2:
        Per-frequency noise variance, ``(F,)`` (currently unused downstream;
        kept for API stability).
    mask:
        Circular real-space mask, ``(H, W)``, applied in the centred projection
        frame before the line integral. Must match the mask used to calibrate
        ``sigma2``. ``None`` projects the full square image.
    frequency_weights:
        Optional per-frequency weights, ``(F,)``, applied before summation
        (e.g. a squared low-pass filter).
    col_batch:
        Number of columns projected together per :func:`jax.lax.map` step.
        Smaller values lower peak memory (allowing larger blocks) at the cost
        of more sequential steps.
    """
    # Pairwise common-line angles; small (n_row * n_col scalars), so the full
    # grid is cheap to hold even when the projections are not.
    angle_row, angle_col, _ = compute_intrinsic_common_line_angles(
        rotations_row[:, None],   # (n_row, 1, 3, 3)
        rotations_col[None, :],   # (1, n_col, 3, 3)
    )  # each (n_row, n_col)

    box = images_row.shape[-1]
    weight = rfft_multiplicity(box)                       # (F,)
    if frequency_weights is not None:
        weight = weight * frequency_weights

    EPS1 = 1e-2
    EPS2 = 1e-8
    
    def one_column(column):
        # Everything specific to a single column particle j.
        image_j, shift_j, ctf_j, angle_row_j, angle_col_j = column

        # Each row image along its pair angle with this column, and this column
        # image along the pair angle for every row.
        lines_row = _project_lines(images_row, shifts_row, angle_row_j, mask)
        lines_col = _project_fan(image_j, shift_j, angle_col_j, mask)  # (n_row, box)

        ft_row = jnp.fft.rfft(lines_row, axis=-1)         # (n_row, F)
        ft_col = jnp.fft.rfft(lines_col, axis=-1)         # (n_row, F)
        delta = ctf_j[None, :] * ft_row - ctf_row * ft_col
        num = jnp.square(delta.real) + jnp.square(delta.imag)
        den = (jnp.square(ctf_j[None,:]) + jnp.square(ctf_row) + EPS1)*sigma2 + EPS2
        return jnp.sum(weight*(num/den), axis=-1)             # (n_row,)

    columns = (images_col, shifts_col, ctf_col, angle_row.T, angle_col.T)
    distances2 = jax.lax.map(one_column, columns, batch_size=col_batch)
    return distances2.T                                   # (n_row, n_col)
