"""Tests for the out-of-core distance driver.

The driver reads through vitrio and keeps several tiles in flight at once, so
the risk worth testing is that the host images a tile still reads are
overwritten under it. Comparing a multi-tile sweep against a single-tile one
catches exactly that.
"""
import numpy as np
import pytest
import mrcfile

import jax
import vitrio as vio

from sinovar import image
from sinovar.ctf import CtfContext
from sinovar.distance import streaming


class _InjectedFailure(Exception):
    """Stands in for whatever the tile kernel might raise (OOM, XLA error)."""


@pytest.fixture
def particles(tmp_path):
    """A small stack on disk, with the poses and CTF the driver needs."""
    count, box = 12, 16
    rng = np.random.default_rng(0)
    path = tmp_path / 'particles.mrcs'
    with mrcfile.new(str(path), overwrite=True) as handle:
        handle.set_data(rng.normal(size=(count, box, box)).astype(np.float32))
        # Without this the header says volume, and indexing into it is an
        # error rather than a stack read.
        handle.set_image_stack()
    return dict(
        locations=image.make_locations(
            [f'{i + 1}@{path.name}' for i in range(count)], str(tmp_path)
        ),
        rotations=np.linalg.qr(rng.normal(size=(count, 3, 3)))[0].astype(
            np.float32),
        shifts=np.zeros((count, 2), np.float32),
        defocus=rng.uniform(8000.0, 20000.0, count).astype(np.float32),
        box_size=box,
        count=count,
    )


def _build(particles, block_size, scratch=None, **kwargs):
    box = particles['box_size']
    return streaming.StreamingSquaredDistanceMatrix(
        loader=vio.loader(workers=2, scratch=scratch),
        image_locations=particles['locations'],
        rotations=particles['rotations'],
        shifts=particles['shifts'],
        defocus=particles['defocus'],
        box_size=box,
        ctf_context=CtfContext(
            pixel_size_a=1.0, voltage_kv=300.0,
            spherical_aberration_mm=2.7, q0=0.07,
        ),
        sigma2=np.full(box // 2 + 1, 0.5, np.float32),
        devices=[jax.devices('cpu')[0]],
        block_size=block_size,
        progress=False,
        **kwargs,
    )


def test_tiling_does_not_change_the_matrix(particles):
    """Several tiles must give what one tile gives.

    A buffer reused too early, or a tile written to the wrong place, shows up
    here and almost nowhere else.
    """
    whole = _build(particles, block_size=particles['count']).run()
    tiled = _build(particles, block_size=4).run()
    np.testing.assert_allclose(tiled, whole, rtol=1e-5, atol=1e-5)


def test_live_blocks_never_share_host_memory(particles):
    """No two blocks alive at once may be backed by the same array.

    Dispatching a tile does not finish with its host images: ``device_put`` is
    asynchronous and on the CPU backend may alias the array rather than copy
    it. A loader that hands back a buffer after dispatch therefore lets a later
    read overwrite a tile still being computed, which showed up as a pipeline
    that gave a different matrix on every run while every unit test passed.
    Timing decides whether that corrupts anything, so the invariant is tested
    directly instead.
    """
    pipeline = _build(particles, block_size=4)
    blocks = streaming._BlockLoader(
        pipeline._loader, pipeline._locations, pipeline._rotations,
        pipeline._shifts, pipeline._ctf,
    )
    live = {}
    for row, col in pipeline._iter_tiles(blocks):
        for block in (row, col):
            live[block.span.index] = block
        for left in live.values():
            for right in live.values():
                if left.span.index == right.span.index:
                    continue
                assert not np.shares_memory(left.images, right.images), (
                    f'blocks {left.span.index} and {right.span.index} share '
                    f'host memory'
                )


def test_matrix_is_symmetric_with_zero_diagonal(particles):
    """The result must be symmetric, exactly so away from the diagonal blocks.

    An off-diagonal tile is written once and mirrored into its transpose, so
    those halves are the same numbers. A diagonal tile computes both halves
    independently, and in float32 the two reductions need not land on the same
    last bit, so it is only symmetric to rounding.
    """
    block = 5
    matrix = _build(particles, block_size=block).run()
    np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-4)

    on_diagonal = np.zeros(matrix.shape, dtype=bool)
    for start in range(0, len(matrix), block):
        stop = min(start + block, len(matrix))
        on_diagonal[start:stop, start:stop] = True

    mirrored = matrix[~on_diagonal]
    np.testing.assert_array_equal(mirrored, matrix.T[~on_diagonal])
    np.testing.assert_allclose(
        matrix[on_diagonal], matrix.T[on_diagonal], rtol=1e-5, atol=1e-5
    )


def test_memory_scratch_changes_nothing(particles):
    """A scratch is a cache: it must not change a single value."""
    plain = _build(particles, block_size=4).run()
    scratch = image.open_scratch(
        'memory', particles['locations'], run_length=4,
    )
    cached = _build(particles, block_size=4, scratch=scratch).run()
    np.testing.assert_allclose(cached, plain, rtol=1e-6, atol=1e-6)


def test_tile_failure_surfaces(particles):
    """A failing tile must raise out of ``run``, not be swallowed or hang."""
    pipeline = _build(particles, block_size=4)
    pipeline._compute_tile = lambda row, col, device: (_ for _ in ()).throw(
        _InjectedFailure('tile kernel failed')
    )
    with pytest.raises(_InjectedFailure):
        pipeline.run()


def test_missing_image_surfaces(particles, tmp_path):
    """A read that cannot be served must raise, with the loader's own error."""
    locations = list(particles['locations'])
    locations[-1] = vio.ImageLocation(str(tmp_path / 'particles.mrcs'), 10_000)
    pipeline = _build({**particles, 'locations': locations}, block_size=4)
    with pytest.raises(Exception) as caught:
        pipeline.run()
    assert not isinstance(caught.value, AssertionError)
