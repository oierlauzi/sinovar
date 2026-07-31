"""The projector's convention must agree with the codebase's common lines.

A theoretical common line is a double projection: the volume onto a particle's
image plane, then that projection onto the common-line axis. The axis of a pair
lies in *both* image planes, so the same line can be reached through either
particle --- and only if the projector's rotation convention is the one
:mod:`sinovar.geometry.common_lines` assumes will the two agree. That
self-consistency is the check these tests are built around; it is the same one
that validated the framework against cryojax.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from sinovar.geometry import (
    compute_intrinsic_common_line_angles,
    euler_zyz_to_matrix,
)
from sinovar.mask import compute_raised_cosine_mask_2d
from sinovar.sinogram.project import _project_image
from sinovar.volume import project_volume, project_volume_basis

_ZERO_SHIFT = jnp.zeros((2,))


def _blob_volume(box_size, n_blobs, rng):
    """A smooth, asymmetric volume, well inside the mask and band-limited.

    Bilinear interpolation is exact for neither, so the test tolerances are set
    by how smooth this is; Gaussians of a few pixels are smooth enough for the
    interpolation error to sit well below the discrepancy a wrong convention
    would cause.
    """
    grid = np.arange(box_size) - 0.5 * (box_size - 1)
    z, y, x = np.meshgrid(grid, grid, grid, indexing='ij')

    sigma = box_size / 24.0
    volume = np.zeros((box_size,) * 3, dtype=np.float32)
    for _ in range(n_blobs):
        centre = rng.uniform(-0.15 * box_size, 0.15 * box_size, size=3)
        distance2 = (
            np.square(x - centre[0])
            + np.square(y - centre[1])
            + np.square(z - centre[2])
        )
        volume += rng.uniform(0.5, 1.5) * np.exp(-0.5 * distance2 / sigma ** 2)

    return volume


def _random_rotations(n, rng):
    """Uniform-ish orientations, through the codebase's own Euler convention."""
    return euler_zyz_to_matrix(
        jnp.asarray(rng.uniform(0.0, 2 * np.pi, size=n)),
        jnp.asarray(np.arccos(rng.uniform(-1.0, 1.0, size=n))),
        jnp.asarray(rng.uniform(0.0, 2 * np.pi, size=n)),
    )


def _projection_mask(box_size):
    return 1.0 - compute_raised_cosine_mask_2d(
        box_size=box_size,
        radius=0.35 * box_size,
        rolloff=0.1 * box_size,
        inside=False,
    )


def _common_line_mismatch(volume, rotation0, rotation1, mask, transpose=False):
    """Relative disagreement of a pair's common line seen from either particle."""
    angle0, angle1, _ = compute_intrinsic_common_line_angles(rotation0, rotation1)

    lines = []
    for rotation, angle in ((rotation0, angle0), (rotation1, angle1)):
        projection = project_volume(volume, rotation.T if transpose else rotation)
        lines.append(_project_image(projection, _ZERO_SHIFT, angle, mask))

    difference = lines[0] - lines[1]
    energy = jnp.sqrt(
        jnp.linalg.norm(lines[0]) * jnp.linalg.norm(lines[1])
    )
    return float(jnp.linalg.norm(difference) / energy)


def test_identity_rotation_sums_along_the_view_axis():
    """The identity samples voxel centres, so the projection is exact."""
    volume = _blob_volume(24, 6, np.random.default_rng(0))

    projection = project_volume(jnp.asarray(volume), jnp.eye(3))

    expected = volume.sum(axis=0)
    np.testing.assert_allclose(
        projection, expected, rtol=1e-5, atol=1e-6 * expected.max()
    )


def test_the_pair_agrees_on_its_common_line():
    rng = np.random.default_rng(1)
    volume = jnp.asarray(_blob_volume(48, 8, rng))
    mask = _projection_mask(48)
    rotations = _random_rotations(2, rng)

    mismatch = _common_line_mismatch(volume, rotations[0], rotations[1], mask)

    assert mismatch < 0.05


def test_transposing_the_rotation_breaks_the_common_line():
    """Pins the convention down: rows are the image axes, not the columns."""
    rng = np.random.default_rng(2)
    volume = jnp.asarray(_blob_volume(48, 8, rng))
    mask = _projection_mask(48)
    rotations = _random_rotations(2, rng)

    mismatch = _common_line_mismatch(volume, rotations[0], rotations[1], mask)
    transposed = _common_line_mismatch(
        volume, rotations[0], rotations[1], mask, transpose=True
    )

    assert transposed > 10 * mismatch


@pytest.mark.parametrize('seed', range(4))
def test_the_pair_agrees_on_its_common_line_for_any_orientation(seed):
    rng = np.random.default_rng(100 + seed)
    volume = jnp.asarray(_blob_volume(48, 8, rng))
    mask = _projection_mask(48)
    rotations = _random_rotations(2, rng)

    assert _common_line_mismatch(volume, rotations[0], rotations[1], mask) < 0.05


def _axis_aligned_pair(in_plane_degrees):
    """A pair whose first common-line angle sits ``in_plane_degrees`` off an axis.

    The geometry is fixed and only image ``i`` is turned in its own plane, which
    moves *its* common-line angle while leaving the pair's common line --- and
    image ``j``'s angle --- exactly where they were. That isolates the projector's
    angular transfer from every other difference between two orientations.
    """
    tilt, in_plane_j = 0.7, 1.3
    # No x component, so the common line d_i x d_j lies along the x axis and
    # image i (viewing down z) reads it out along a grid row.
    direction = np.array([0.0, np.cos(tilt), np.sin(tilt)])
    axis = np.array([1.0, 0.0, 0.0])
    other = np.cross(direction, axis)
    rotation_j = np.stack([
        np.cos(in_plane_j) * axis + np.sin(in_plane_j) * other,
        -np.sin(in_plane_j) * axis + np.cos(in_plane_j) * other,
        direction,
    ])

    phi = np.radians(in_plane_degrees)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    rotation_i = np.array([
        [cos_phi, sin_phi, 0.0], [-sin_phi, cos_phi, 0.0], [0.0, 0.0, 1.0],
    ])
    return jnp.asarray(rotation_i), jnp.asarray(rotation_j)


def test_an_axis_aligned_common_line_is_the_worst_case():
    """Bilinear resampling is exact along a grid axis and smooths away from it.

    A line on an axis is read straight off the samples while its partner at a
    general angle is interpolated, so the pair disagrees by more than two generic
    angles do --- this is where a projector-dependent error comes from. Real
    orientations never land exactly there and the excess decays within a few
    degrees, but it bounds what the projector contributes to the model.
    """
    rng = np.random.default_rng(5)
    volume = jnp.asarray(_blob_volume(48, 8, rng))
    mask = _projection_mask(48)

    aligned = _common_line_mismatch(volume, *_axis_aligned_pair(0.0), mask)
    generic = _common_line_mismatch(volume, *_axis_aligned_pair(20.0), mask)

    assert aligned < 0.05, f'the worst case is worse than expected: {aligned}'
    assert aligned > 2.0 * generic, 'the axis is meant to be the worst case'


def test_the_slab_size_only_groups_the_integration():
    """A memory knob must not change the result, including when it does not divide."""
    rng = np.random.default_rng(3)
    volume = jnp.asarray(_blob_volume(30, 5, rng))
    rotation = _random_rotations(1, rng)[0]

    reference = project_volume(volume, rotation, slab=1)

    for slab in (4, 7, 30):
        np.testing.assert_allclose(
            project_volume(volume, rotation, slab=slab), reference, atol=1e-3
        )


def test_a_basis_projects_every_volume():
    rng = np.random.default_rng(4)
    basis = jnp.asarray(np.stack([_blob_volume(24, 4, rng) for _ in range(3)]))
    rotation = _random_rotations(1, rng)[0]

    projections = project_volume_basis(basis, rotation)

    assert projections.shape == (3, 24, 24)
    for index, volume in enumerate(basis):
        np.testing.assert_allclose(
            projections[index], project_volume(volume, rotation), atol=1e-4
        )


def test_rejects_a_volume_that_is_not_a_basis():
    with pytest.raises(ValueError):
        project_volume_basis(jnp.zeros((8, 8, 8)), jnp.eye(3))
