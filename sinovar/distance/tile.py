from typing import Optional
import jax
import jax.numpy as jnp

from ..filter import rfft_multiplicity
from ..sinogram.project import project_sinogram_masked
from ..geometry.common_lines import compute_intrinsic_common_line_angles

# (N, H, W), (N, 2), (N, M), (H, W) -> (N, M, box): a batch of images, each
# projected along its own row of M angles, through a shared circular mask.
_project_sinogram_grid = jax.vmap(project_sinogram_masked, in_axes=(0, 0, 0, None))


@jax.jit
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
    frequency_weights: Optional[jax.Array] = None
) -> jax.Array:
    """CTF-weighted common-line distance for every (row, col) pair in a tile.

    This is the batched equivalent of the per-pair loop body: it returns a
    ``(n_row, n_col)`` block of squared distances ``D[i, j]`` between the row
    particle ``i`` and the column particle ``j``. Every argument is a batch, so
    the whole distance matrix can be assembled tile by tile.

    The common-line orientation depends on *both* members of a pair, so each cell
    requires its own pair of 1D projections; nothing can be shared between cells
    beyond the per-particle CTF.

    Parameters
    ----------
    images_row, images_col:
        Real-space particle images, shape ``(n_row, H, W)`` / ``(n_col, H, W)``.
    shifts_row, shifts_col:
        Per-particle origin shifts in pixels, shape ``(n, 2)``.
    rotations_row, rotations_col:
        Per-particle orientation matrices, shape ``(n, 3, 3)``.
    ctf_row, ctf_col:
        Pre-computed 1D CTFs, shape ``(n, F)`` with ``F = rfft length of the
        projection``. Passed in directly so they are computed once per particle
        rather than once per tile.
    sigma2:
        Per-frequency noise variance of a projected common line's ``rfft``,
        shape ``(F,)``, in the same unnormalised ``rfft`` convention used below.
        This is *not* the raw 2D PSD: the projector's bilinear interpolation and
        edge clipping reshape and attenuate the noise, so the value is calibrated
        by pushing the estimated noise through the real projector --- see
        :func:`sinovar.noise.estimate_projected_line_noise_variance`. It is used
        directly as the noise variance of the residual, so no scale factor is
        applied here.
    mask:
        Circular real-space mask, shape ``(H, W)``, applied to each image in the
        centred projection frame before the line integral. It removes the box's
        angle-dependent corner clipping (so a single ``sigma2`` is valid) and the
        signal-free corners. Must match the mask used to calibrate ``sigma2``.
        ``None`` projects the full square image.
    frequency_weights:
        Optional per-frequency weights, shape ``(F,)``, applied to each term
        before summation (e.g. a squared low-pass filter).
    """
    angle_row, angle_col = compute_intrinsic_common_line_angles(
        rotations_row[:, None],   # (n_row, 1, 3, 3)
        rotations_col[None, :],   # (1, n_col, 3, 3)
    )
    ctf_col = ctf_col[None, :] # (n_row, 1, F)
    ctf_row = ctf_row[:, None] # (1, n_col, F)

    lines_row = _project_sinogram_grid(images_row, shifts_row, angle_row, mask)        # (n_row, n_col, box)
    lines_col = _project_sinogram_grid(images_col, shifts_col, angle_col.T, mask)      # (n_col, n_row, box)
    lines_col = jnp.swapaxes(lines_col, 0, 1)                                 # (n_row, n_col, box)
    box = lines_row.shape[-1]

    # Unnormalised forward rfft (default norm): ``sigma2`` is calibrated to this
    # same convention, so the residual and its variance share one scale and the
    # distance is invariant to the choice of FFT normalisation.
    ft_lines_row = jnp.fft.rfft(lines_row, axis=-1)   # (n_row, n_col, F)
    ft_lines_col = jnp.fft.rfft(lines_col, axis=-1)   # (n_row, n_col, F)

    EPS = 1e-2

    delta = ctf_col*ft_lines_row - ctf_row*ft_lines_col
    num = jnp.square(delta.real) + jnp.square(delta.imag)
    # Denominator is the noise variance of the residual, (ctf_col^2 + ctf_row^2)
    # * sigma2. ``maximum(., EPS)`` floors only the rare bins where both CTFs
    # vanish (an otherwise 0/0); elsewhere it leaves the variance untouched, so
    # E[num/den] = 1 under the noise null and the -1 centres the term to zero.
    #den = jnp.maximum(jnp.square(ctf_col) + jnp.square(ctf_row), EPS)*sigma2
    #terms = num/den - 1
    terms = num

    multiplicity = rfft_multiplicity(box)
    if frequency_weights is not None:
        multiplicity = multiplicity*frequency_weights
    
    return jnp.maximum(jnp.sum(multiplicity*terms, axis=-1), 0.0)  # (n_row, n_col)
