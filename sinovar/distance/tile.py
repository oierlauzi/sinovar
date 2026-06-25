import jax
import jax.numpy as jnp

# `_project_image` is the single-image building block behind `sinogram.project_images`.
# We reuse it directly (instead of the public batched `project_images`) so that we can
# nest the vmaps with `in_axes=None` on the image: each image stays resident on the
# device and is projected at many angles without ever materialising one image copy
# per angle.
from ..sinogram.project import _project_image
from ..geometry.common_lines import compute_intrinsic_common_line_angles

# (H, W), (2,), (M,) -> (M, box): one image projected along M different lines.
_project_image_multi_angle = jax.vmap(_project_image, in_axes=(None, None, 0))
# (N, H, W), (N, 2), (N, M) -> (N, M, box): a batch of images, each projected along
# its own row of M angles.
_project_image_grid = jax.vmap(_project_image_multi_angle, in_axes=(0, 0, 0))


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
    """
    # angle_row[i, j] / angle_col[i, j]: intrinsic angle of the common line shared
    # by the orientations of row particle i and column particle j, expressed in each
    # particle's own frame.
    angle_row, angle_col = compute_intrinsic_common_line_angles(
        rotations_row[:, None],   # (n_row, 1, 3, 3)
        rotations_col[None, :],   # (1, n_col, 3, 3)
    )  # both -> (n_row, n_col)

    # lines_row[i, j] = projection of row image i along the (i, j) common line.
    lines_row = _project_image_grid(images_row, shifts_row, angle_row)        # (n_row, n_col, box)
    # lines_col[j, i] = projection of col image j along the (i, j) common line;
    # transpose back so the pair axes line up with `lines_row`.
    lines_col = _project_image_grid(images_col, shifts_col, angle_col.T)      # (n_col, n_row, box)
    lines_col = jnp.swapaxes(lines_col, 0, 1)                                 # (n_row, n_col, box)

    ft_row = jnp.fft.rfft(lines_row, axis=-1)   # (n_row, n_col, F)
    ft_col = jnp.fft.rfft(lines_col, axis=-1)   # (n_row, n_col, F)

    # delta[i, j] = ctf_col[j] * ft_row[i, j] - ctf_row[i] * ft_col[i, j]
    delta = ctf_col[None, :, :] * ft_row - ctf_row[:, None, :] * ft_col
    return jnp.sum(jnp.square(delta.real) + jnp.square(delta.imag), axis=-1)  # (n_row, n_col)
