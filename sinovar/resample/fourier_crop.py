"""Band-limited resampling by cropping (or padding) the discrete Fourier transform.

The ``refine`` objective only compares images with the eigen-volume basis up to
the resolution it is asked for, while projecting a volume costs ``box^3`` and
extracting a common line ``box^2``. Throwing the frequencies above that
resolution away *before* any of that work happens is therefore the single
largest saving available, and cropping the DFT is the exact way to do it: every
retained coefficient is left untouched, so the result is the band-limited
original sampled on a coarser grid rather than an approximation of it.

Images and volumes are resampled the same way and to the same box, which keeps
them on a common grid: the physical extent ``box_size * pixel_size`` is
preserved, so a frequency bin index means the same thing in both and the CTF,
the noise spectrum and the mask need only be rebuilt at the new box size.

The scaling is chosen to preserve *values* rather than sums --- a constant array
resamples to the same constant --- so a line integral over the resampled grid
picks up the sampling factor ``box_new / box_old``. That factor is shared by
every image and every volume, so it cancels in any comparison between them.

One asymmetry is unavoidable when cropping to an even box: the retained block
holds the ``-box/2`` frequency but not its ``+box/2`` partner, so the inverse
transform has a small imaginary part, which is discarded. Its size is that of
the signal at the cut-off, which is why ``refine`` crops well above the
resolution its low-pass filter actually keeps.
"""
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=('box_size', 'n_spatial_dims'))
def fourier_crop(
    x: jax.Array,
    box_size: int,
    n_spatial_dims: int
) -> jax.Array:
    """Resample the trailing ``n_spatial_dims`` axes of ``x`` to ``box_size``.

    Parameters
    ----------
    x:
        Real array whose trailing ``n_spatial_dims`` axes are all the same
        length; any leading axes are treated as a batch.
    box_size:
        Target length of each trailing axis. Smaller than the input crops the
        transform (a band limit), larger pads it with zeros (an interpolation).
    n_spatial_dims:
        Number of trailing axes to resample: 2 for images, 3 for volumes.
    """
    source = x.shape[-1]
    if any(dimension != source for dimension in x.shape[-n_spatial_dims:]):
        raise ValueError(
            f'the trailing {n_spatial_dims} axes must be equal, got '
            f'{x.shape[-n_spatial_dims:]}'
        )
    if box_size == source:
        return x

    axes = tuple(range(x.ndim - n_spatial_dims, x.ndim))
    batch = x.ndim - n_spatial_dims

    # Centre the transform so that the block of retained frequencies is
    # contiguous; ``fftshift`` puts DC at ``length // 2``, which is where the
    # resampled array expects it too.
    ft = jnp.fft.fftshift(jnp.fft.fftn(x, axes=axes), axes=axes)
    if box_size < source:
        start = source // 2 - box_size // 2
        selection = (
            (slice(None),) * batch
            + (slice(start, start + box_size),) * n_spatial_dims
        )
        ft = ft[selection]
    else:
        before = box_size // 2 - source // 2
        padding = (
            ((0, 0),) * batch
            + ((before, box_size - source - before),) * n_spatial_dims
        )
        ft = jnp.pad(ft, padding)

    resampled = jnp.fft.ifftn(jnp.fft.ifftshift(ft, axes=axes), axes=axes)
    gain = (box_size / source) ** n_spatial_dims
    return gain * resampled.real


def fourier_crop_2d(images: jax.Array, box_size: int) -> jax.Array:
    """Resample a stack of images, ``(..., H, W) -> (..., box_size, box_size)``."""
    return fourier_crop(images, box_size, 2)


def fourier_crop_3d(volumes: jax.Array, box_size: int) -> jax.Array:
    """Resample volumes, ``(..., D, H, W) -> (..., box_size, box_size, box_size)``."""
    return fourier_crop(volumes, box_size, 3)
