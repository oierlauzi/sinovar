import jax
import jax.numpy as jnp

from ..filter import rfft_multiplicity


@jax.jit
def estimate_noise_psd_profile(
    images: jax.Array,
    outside_mask: jax.Array
) -> jax.Array:
    masked_images = images*outside_mask
    masked_images_ft = jnp.fft.rfft2(masked_images)
    masked_images_spectra = (
        jnp.square(masked_images_ft.real) +
        jnp.square(masked_images_ft.imag)
    )
    mask_energy = jnp.sum(jnp.square(outside_mask))

    batch_axes = tuple(range(images.ndim - 2))
    spectrum = jnp.mean(masked_images_spectra, axis=batch_axes) / mask_energy

    box_size = spectrum.shape[-2]
    n_bins = box_size // 2 + 1

    fy = jnp.fft.fftfreq(box_size) * box_size
    fx = jnp.fft.rfftfreq(box_size) * box_size
    radius = jnp.hypot(fy[:, None], fx[None, :])
    bin_index = jnp.round(radius).astype(jnp.int32).ravel()

    weights = jnp.broadcast_to(rfft_multiplicity(box_size), spectrum.shape)

    radial_sum = jnp.bincount(
        bin_index, weights=(weights*spectrum).ravel(), length=n_bins
    )
    radial_weight = jnp.bincount(
        bin_index, weights=weights.ravel(), length=n_bins
    )
    return radial_sum / radial_weight
