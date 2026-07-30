"""Cropping the transform must be lossless on whatever it does not discard.

The one thing to keep in mind while reading these: an even box holds the
``-box/2`` frequency but not its ``+box/2`` partner, so a crop whose boundary
carries energy is not quite reversible --- the discarded imaginary part is that
asymmetry. The tests therefore work with band-limited signals, which is also the
only regime the ``refine`` command resamples in.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from sinovar.resample import fourier_crop, fourier_crop_2d, fourier_crop_3d


def _band_limited(box_size, n_spatial_dims, radius, rng):
    """A signal whose spectrum vanishes outside ``radius``, so a crop is lossless."""
    frequency = np.fft.fftfreq(box_size) * box_size
    grid = np.meshgrid(*([frequency] * n_spatial_dims), indexing='ij')
    inside = np.sqrt(sum(np.square(axis) for axis in grid)) <= radius

    shape = (box_size,) * n_spatial_dims
    spectrum = (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ) * inside
    return np.fft.ifftn(spectrum).real.astype(np.float32)


def test_a_constant_resamples_to_the_same_constant():
    """The scaling preserves values, not sums."""
    images = jnp.full((3, 32, 32), 2.5)

    np.testing.assert_allclose(fourier_crop_2d(images, 8), 2.5, rtol=1e-5)


def test_the_target_box_is_a_no_op():
    images = jnp.asarray(_band_limited(16, 2, 4, np.random.default_rng(0)))

    np.testing.assert_array_equal(fourier_crop_2d(images, 16), images)


def test_a_band_limited_image_survives_a_round_trip():
    """Cropping only discards what a band-limited image does not have."""
    image = jnp.asarray(_band_limited(64, 2, 6, np.random.default_rng(1)))

    cropped = fourier_crop_2d(image, 32)
    restored = fourier_crop_2d(cropped, 64)

    assert cropped.shape == (32, 32)
    np.testing.assert_allclose(restored, image, atol=1e-5)


def test_cropping_keeps_the_frequencies_below_the_band_untouched():
    rng = np.random.default_rng(2)
    image = jnp.asarray(_band_limited(32, 2, 6, rng))

    cropped = fourier_crop_2d(image, 16)

    # On the centred grid the crop is a plain slice, so the retained
    # coefficients must survive it bar the value-preserving scaling.
    reference = np.fft.fftshift(np.fft.fft2(np.asarray(image)))[8:24, 8:24]
    got = np.fft.fftshift(np.fft.fft2(np.asarray(cropped)))
    np.testing.assert_allclose(got, reference / 4.0, atol=1e-4)


def test_a_batch_of_volumes_keeps_its_leading_axes():
    volumes = jnp.zeros((2, 5, 12, 12, 12))

    assert fourier_crop_3d(volumes, 6).shape == (2, 5, 6, 6, 6)


def test_padding_interpolates_a_volume():
    rng = np.random.default_rng(3)
    volume = jnp.asarray(_band_limited(8, 3, 2.5, rng))

    upsampled = fourier_crop_3d(volume, 16)

    assert upsampled.shape == (16, 16, 16)
    # Padding adds no information, so cropping back is exact.
    np.testing.assert_allclose(fourier_crop_3d(upsampled, 8), volume, atol=1e-5)


def test_rejects_axes_that_are_not_square():
    with pytest.raises(ValueError):
        fourier_crop(jnp.zeros((8, 4)), 4, 2)
