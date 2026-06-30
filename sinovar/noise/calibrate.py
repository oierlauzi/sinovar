from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp

from ..sinogram.project import project_sinogram_masked

# A batch of images, one random angle each; the shift (zero here) and the mask
# are shared across the batch. A translation only adds a phase ramp and leaves
# |rfft|^2 unchanged, so projecting with zero shift is sufficient.
_project_lines = jax.vmap(project_sinogram_masked, in_axes=(0, None, 0, None))


@partial(jax.jit, static_argnames=("box_size", "n_samples"))
def estimate_projected_line_noise_variance(
    noise_spectra: jax.Array,
    box_size: int,
    key: jax.Array,
    mask: Optional[jax.Array] = None,
    n_samples: int = 1024,
) -> jax.Array:
    """Per-frequency noise variance of a common-line projection's rfft.

    :func:`estimate_noise_psd_profile` measures the noise power per *2D* Fourier
    mode, but the common-line distance compares *1D* projections produced by
    :func:`project_sinogram_masked`. That projector samples the image with
    bilinear interpolation, which reshapes and attenuates the noise spectrum.
    Rather than model that transfer analytically, this pushes synthetic noise
    carrying the estimated PSD through the *same* projector (with the *same*
    mask) and measures the resulting per-frequency variance --- exactly the
    ``sigma2`` the distance kernel needs, with no scale factor left to guess.

    The circular ``mask`` is what makes a single per-frequency ``sigma2`` valid:
    it removes the square box's angle-dependent corner clipping, so the line
    noise variance no longer depends on the (generic) projection angle and need
    not be tabulated per angle.

    Parameters
    ----------
    noise_spectra:
        Radial per-pixel noise PSD, shape ``(box_size // 2 + 1,)``, e.g. the
        output of :func:`estimate_noise_psd_profile`.
    box_size:
        Side length of the (square) particle image.
    key:
        PRNG key for the synthetic noise and its projection angles.
    mask:
        Circular real-space mask applied before each projection, shape
        ``(box_size, box_size)``; must match the mask used by the distance
        kernel. ``None`` reproduces the unmasked (angle-dependent) behaviour.
    n_samples:
        Number of noise realisations to average; the estimate is a Monte-Carlo
        mean, so more samples give a smoother ``sigma2``.

    Returns
    -------
    jax.Array
        Per-frequency line-FT noise variance, shape ``(box_size // 2 + 1,)``,
        in the unnormalised ``rfft`` convention used by ``compute_distance2_tile``.
    """
    # Lift the radial profile to a 2D PSD: the inverse of the radial binning in
    # estimate_noise_psd_profile, so the synthetic noise reproduces the estimate.
    fy = jnp.fft.fftfreq(box_size) * box_size
    fx = jnp.fft.rfftfreq(box_size) * box_size
    radius = jnp.hypot(fy[:, None], fx[None, :])
    n_bins = noise_spectra.shape[-1]
    bin_index = jnp.clip(jnp.round(radius).astype(jnp.int32), 0, n_bins - 1)
    psd_2d = noise_spectra[bin_index]

    # Unit white noise has E|rfft2|^2 = box^2 (unnormalised rfft2), matching the
    # convention of estimate_noise_psd_profile, so scaling by sqrt(psd_2d)
    # synthesises real images whose spectrum is exactly the estimated PSD.
    key_white, key_angle = jax.random.split(key)
    white = jax.random.normal(key_white, (n_samples, box_size, box_size))
    images = jnp.fft.irfft2(
        jnp.fft.rfft2(white) * jnp.sqrt(psd_2d), s=(box_size, box_size)
    )

    # Project through the real operator at random angles and measure the variance
    # of the default-norm rfft frequency by frequency. With the circular mask the
    # variance is angle-independent, so averaging over uniform angles is unbiased.
    angles = jax.random.uniform(key_angle, (n_samples, 1), maxval=2 * jnp.pi)
    shift = jnp.zeros((2,), dtype=images.dtype)
    lines = _project_lines(images, shift, angles, mask)[:, 0, :]   # (n_samples, box)
    ft = jnp.fft.rfft(lines, axis=-1)
    return jnp.mean(jnp.square(ft.real) + jnp.square(ft.imag), axis=0)
