"""End-to-end checks of the common-line refinement on synthetic data.

The synthetic dataset is built through the *forward* model the refinement
inverts: every particle is a real-space projection of ``mean + U z_i`` at its own
orientation. Nothing is idealised --- the measurements come out of the same
crossed-CTF comparison of two experimental lines that the refinement measures,
with the model's line taken from the anchor's basis projections --- so a correct
implementation must recover ``z`` up to the interpolation floor of the projector
(a fraction of a percent, see :mod:`tests.test_volume`) and the blindness of the
common lines actually sampled.
"""
import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sinovar.geometry import euler_zyz_to_matrix
from sinovar.mask import compute_raised_cosine_mask_2d
from sinovar.refine import refine_embedding
from sinovar.refine.refine import _calibrate
from sinovar.sinogram.project import _project_image
from sinovar.volume import project_volume

_BOX = 32
_COMPONENTS = 3
_PARTICLES = 40


def _blob_volume(box_size, n_blobs, rng, signed=False):
    """A smooth volume well inside the mask, so the projector interpolates it well."""
    grid = np.arange(box_size) - 0.5 * (box_size - 1)
    z, y, x = np.meshgrid(grid, grid, grid, indexing='ij')

    sigma = box_size / 10.0
    volume = np.zeros((box_size,) * 3, dtype=np.float32)
    for _ in range(n_blobs):
        centre = rng.uniform(-0.1 * box_size, 0.1 * box_size, size=3)
        distance2 = (
            np.square(x - centre[0])
            + np.square(y - centre[1])
            + np.square(z - centre[2])
        )
        amplitude = rng.uniform(-1.5, 1.5) if signed else rng.uniform(0.5, 1.5)
        volume += amplitude * np.exp(-0.5 * distance2 / sigma ** 2)

    return volume


def _projection_mask(box_size):
    return 1.0 - compute_raised_cosine_mask_2d(
        box_size=box_size,
        radius=0.4 * box_size,
        rolloff=0.1 * box_size,
        inside=False,
    )


def _dataset(seed=0):
    """A dataset whose embedding is known, plus everything ``refine`` needs."""
    rng = np.random.default_rng(seed)

    # A signed basis, so the eigen-volumes are not all the same broad bump and
    # the pair metrics are not near-degenerate.
    volumes = np.stack([
        _blob_volume(_BOX, 4, rng, signed=True) for _ in range(_COMPONENTS)
    ])
    mean = _blob_volume(_BOX, 8, rng)
    embedding = rng.normal(size=(_PARTICLES, _COMPONENTS))

    rotations = euler_zyz_to_matrix(
        jnp.asarray(rng.uniform(0.0, 2 * np.pi, size=_PARTICLES)),
        jnp.asarray(np.arccos(rng.uniform(-1.0, 1.0, size=_PARTICLES))),
        jnp.asarray(rng.uniform(0.0, 2 * np.pi, size=_PARTICLES)),
    )

    images = np.stack([
        np.asarray(project_volume(
            jnp.asarray(mean + np.tensordot(embedding[index], volumes, axes=1)),
            rotations[index],
        ))
        for index in range(_PARTICLES)
    ])

    n_frequencies = _BOX // 2 + 1
    return embedding, dict(
        volumes=volumes,
        images=images,
        rotations=np.asarray(rotations),
        shifts=np.zeros((_PARTICLES, 2), dtype=np.float32),
        # A flat CTF keeps the crossed weighting exact and the forward model
        # trivially invertible; the CTF itself is exercised by test_distance.
        ctf=np.ones((_PARTICLES, n_frequencies), dtype=np.float32),
        noise_variance=np.zeros((n_frequencies,), dtype=np.float32),
        mask=np.asarray(_projection_mask(_BOX)),
    )


def _error(embedding, truth):
    """RMS coordinate error relative to the RMS radius, up to a translation."""
    embedding = embedding - embedding.mean(axis=0)
    truth = truth - truth.mean(axis=0)
    return float(
        np.sqrt(np.mean(np.sum(np.square(embedding - truth), axis=-1)))
        / np.sqrt(np.mean(np.sum(np.square(truth), axis=-1)))
    )


def _refine(initial, data, **kwargs):
    options = dict(
        n_epochs=8,
        anchors_per_step=4,
        partners_per_anchor=_PARTICLES - 1,
        ridge=1e-3,
        scale=1.0,
        n_scale_steps=2,
        progress=False,
    )
    options.update(kwargs)
    return refine_embedding(initial, **data, **options)


def test_the_truth_is_a_fixed_point():
    """With the measurements matched, every pull vanishes and nothing moves."""
    truth, data = _dataset()

    refined = _refine(truth.copy(), data)

    assert _error(refined, truth) < 0.05


def test_a_perturbed_embedding_is_pulled_back():
    truth, data = _dataset(seed=1)
    initial = truth + np.random.default_rng(2).normal(scale=0.4, size=truth.shape)

    refined = _refine(initial, data)

    assert _error(refined, truth) < 0.5 * _error(initial, truth)


def test_the_fitted_scale_leaves_a_matching_embedding_alone():
    """The fit must land on the 1 the dataset was built with.

    Any other value would make every modelled distance uniformly wrong and the
    embedding would be pulled off the truth to compensate.
    """
    truth, data = _dataset(seed=3)

    refined = _refine(truth.copy(), data, scale=None, n_scale_steps=4)

    assert _error(refined, truth) < 0.05


def test_a_perturbation_is_undone_with_the_scale_fitted():
    """The fit pins the units; the pairs still fix the shape within them."""
    truth, data = _dataset(seed=3)
    initial = truth + np.random.default_rng(9).normal(scale=0.4, size=truth.shape)

    refined = _refine(initial, data, scale=None, n_scale_steps=4)

    assert _error(refined, truth) < 0.6 * _error(initial, truth)


def test_the_fitted_scale_absorbs_a_uniformly_scaled_embedding():
    """The flip side: a global scale error in the input is *not* what this fixes.

    ``(s / a, a z)`` models the data exactly as well as ``(s, z)``, so fitting
    the scale from the input embedding keeps whatever units it arrived in and
    only its shape is refined. Pass ``scale`` explicitly to pin the units
    elsewhere.
    """
    truth, data = _dataset(seed=3)

    fitted = _refine(0.6 * truth, data, scale=None, n_scale_steps=4)
    given = _refine(0.6 * truth, data, scale=1.0)

    assert _error(fitted, 0.6 * truth) < 0.05
    assert _error(given, truth) < 0.15


def _ctf(frequency2, defocus, wavelength=0.0197, spherical_aberration=2.7e7):
    """The library's CTF, evaluable on a 1D or a 2D frequency-squared grid.

    Sharing one expression between the images and the ``ctf`` handed to the
    refinement is what makes the 1D profile along a common line equal the 2D
    transfer the image carries --- true only because there is no astigmatism.
    """
    angle = np.pi * wavelength * frequency2 * (
        0.5 * spherical_aberration * wavelength ** 2 * frequency2 - defocus
    )
    return -np.sin(angle - np.arctan(0.1 / np.sqrt(1.0 - 0.01)))


def _line_noise_variance(noise_std, mask, rng, n_samples=512):
    """Per-frequency variance of a projected line, by Monte Carlo.

    Measured rather than derived, and independently of
    :mod:`sinovar.noise`, so that the calibration is tested against the
    variance the images actually have.
    """
    images = jnp.asarray(rng.normal(scale=noise_std, size=(n_samples, _BOX, _BOX)))
    angles = jnp.asarray(rng.uniform(0.0, 2 * np.pi, size=n_samples))
    lines = jax.vmap(_project_image, in_axes=(0, None, 0, None))(
        images, jnp.zeros((2,)), angles, jnp.asarray(mask)
    )
    spectrum = jnp.fft.rfft(lines, axis=-1)
    return np.asarray(jnp.mean(
        jnp.square(spectrum.real) + jnp.square(spectrum.imag), axis=0
    ))


def _observed(data, seed, noise_fraction=0.5):
    """Push a clean dataset through the forward model: CTF, then additive noise.

    Returns the true per-frequency line noise variance alongside, which is what
    the calibration is asked to recover.
    """
    rng = np.random.default_rng(seed)
    clean = data['images']

    # Chosen so the transfer has a couple of oscillations across the band and is
    # therefore resolved by its rfft samples: only then does the 1D profile along
    # a common line describe the 2D transfer the image carries. (Real data is
    # comfortably in this regime; a small box with a large defocus is not.)
    defocus = rng.uniform(200.0, 800.0, size=_PARTICLES)
    k = np.fft.fftfreq(_BOX)
    k_half = np.fft.rfftfreq(_BOX)
    frequency2_2d = np.square(k)[:, None] + np.square(k_half)[None, :]
    data['ctf'] = _ctf(
        np.square(k_half)[None, :], defocus[:, None]
    ).astype(np.float32)
    filtered = np.fft.irfft2(
        np.fft.rfft2(clean) * _ctf(frequency2_2d[None], defocus[:, None, None]),
        s=(_BOX, _BOX),
    )

    noise_std = noise_fraction * filtered.std()
    data['images'] = (
        filtered + rng.normal(scale=noise_std, size=filtered.shape)
    ).astype(np.float32)
    data['noise_variance'] = _line_noise_variance(
        noise_std, data['mask'], rng
    ).astype(np.float32)
    return data


def _fitted_floor_scale(caplog):
    """The ``beta`` the calibration settled on, out of the log it reports."""
    message = next(
        record.getMessage() for record in caplog.records
        if 'noise floor x' in record.getMessage()
    )
    return float(message.split('noise floor x')[1].split()[0])


def test_the_noise_floor_is_calibrated_against_the_data(caplog):
    """A wrong sigma2 must be corrected by beta rather than absorbed by z.

    The dataset carries a known noise level, and the refinement is handed a
    variance four times too small --- the bias that a mask-corner estimate makes
    when the background under the particle is stronger. The fit has to find it.
    """
    truth, data = _dataset(seed=10)
    data = _observed(data, seed=11)
    data['noise_variance'] = 0.25 * data['noise_variance']

    with caplog.at_level(logging.INFO, logger='sinovar.refine.refine'):
        _refine(truth.copy(), data, scale=None, n_scale_steps=4)

    floor_scale = _fitted_floor_scale(caplog)
    assert 3.0 < floor_scale < 5.5, f'recovered {floor_scale}, expected about 4'


def test_the_calibration_recovers_a_correct_noise_floor_as_one(caplog):
    """The other side of it: a right sigma2 must be left alone."""
    truth, data = _dataset(seed=12)
    data = _observed(data, seed=13)

    with caplog.at_level(logging.INFO, logger='sinovar.refine.refine'):
        _refine(truth.copy(), data, scale=None, n_scale_steps=4)

    floor_scale = _fitted_floor_scale(caplog)
    assert 0.7 < floor_scale < 1.4, f'a correct floor was rescaled to {floor_scale}'


def test_a_given_floor_scale_is_not_fitted(caplog):
    """The override has to survive the calibration, not be overwritten by it."""
    truth, data = _dataset(seed=14)
    data = _observed(data, seed=15)
    data['noise_variance'] = 0.25 * data['noise_variance']

    with caplog.at_level(logging.INFO, logger='sinovar.refine.refine'):
        _refine(truth.copy(), data, scale=None, n_scale_steps=4, floor_scale=2.5)

    assert _fitted_floor_scale(caplog) == pytest.approx(2.5)


def test_the_calibration_falls_back_when_the_regressors_are_collinear(caplog):
    """Only the sum of the two terms is determined, so beta must be left alone."""
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    floor = 3.0 * signal          # proportional across pairs: nothing to separate
    measured2 = 2.0 * signal + 3.0 * floor
    totals = {
        'measured2': measured2.sum(), 'floor': floor.sum(),
        'qq': (signal * signal).sum(), 'qc': (signal * floor).sum(),
        'cc': (floor * floor).sum(),
        'q_measured2': (signal * measured2).sum(),
        'c_measured2': (floor * measured2).sum(),
    }

    with caplog.at_level(logging.WARNING, logger='sinovar.refine.refine'):
        scale2, floor_scale, correlation = _calibrate(
            totals, 1.0, fit_scale=True, fit_floor=True
        )

    assert floor_scale == 1.0
    assert correlation == pytest.approx(1.0)
    assert any('proportional' in record.getMessage() for record in caplog.records)


def test_the_centroid_and_the_component_order_survive():
    truth, data = _dataset(seed=4)
    initial = truth + 5.0

    refined = _refine(initial, data, n_epochs=2)

    np.testing.assert_allclose(refined.mean(axis=0), initial.mean(axis=0), atol=1e-4)
    # Component k is eigen-volume k, so the axes must not be rotated or reordered
    # the way an MDS embedding's are.
    assert _error(refined, truth) < 0.2


def test_a_single_epoch_already_improves_the_fit():
    truth, data = _dataset(seed=5)
    initial = 1.5 * truth

    once = _refine(initial, data, n_epochs=1)

    assert _error(once, truth) < 0.7 * _error(initial, truth)


def test_the_batching_knobs_do_not_change_the_result():
    truth, data = _dataset(seed=6)

    reference = _refine(truth.copy(), data, n_epochs=1)
    batched = _refine(
        truth.copy(), data, n_epochs=1, partner_batch=3, anchor_batch=2, slab=5
    )

    np.testing.assert_allclose(batched, reference, atol=1e-3)


@pytest.mark.parametrize('kwargs', [
    dict(n_epochs=0),
    dict(anchors_per_step=0),
    dict(partners_per_anchor=0),
    dict(ridge=0.0),
    dict(huber=0.0),
    dict(step_size=0.0),
    dict(step_size=2.5),
    dict(scale=0.0),
    dict(floor_scale=0.0),
])
def test_rejects_out_of_range_arguments(kwargs):
    truth, data = _dataset(seed=7)
    with pytest.raises(ValueError):
        _refine(truth, data, **kwargs)


@pytest.mark.parametrize('key, shape', [
    ('volumes', (_COMPONENTS, _BOX, _BOX, _BOX - 2)),
    ('images', (_PARTICLES, _BOX, _BOX - 2)),
    ('rotations', (_PARTICLES, 3, 2)),
    ('shifts', (_PARTICLES, 3)),
    ('ctf', (_PARTICLES, _BOX)),
    ('noise_variance', (_BOX,)),
    ('mask', (_BOX, _BOX - 1)),
])
def test_rejects_mismatched_arrays(key, shape):
    truth, data = _dataset(seed=8)
    data[key] = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError):
        _refine(truth, data)
