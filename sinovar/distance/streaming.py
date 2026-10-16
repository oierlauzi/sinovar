from typing import Iterator, List, Optional, Sequence, Tuple
from collections import deque
from dataclasses import dataclass
import logging

import numpy as np
import jax
import jax.numpy as jnp
import vitrio as vio
from tqdm import tqdm

from .tile import compute_distance2_tile
from ..filter import butterworth_1d
from ..ctf import CtfContext, compute_ctf_1d

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Span:
    """Which particles a block covers."""

    index: int
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class _Block:
    """A span of particles resident in one of the host buffers."""

    span: _Span
    images: np.ndarray
    rotations: np.ndarray
    shifts: np.ndarray
    ctf: np.ndarray


class _BlockLoader:
    """Reads contiguous spans of particles out of vitrio, one array per span.

    A request returns a completion straight away, so the caller can start the
    next span before it needs this one, and vitrio does the reading on its own
    worker pool.

    Each request allocates its own destination rather than drawing from a pool
    of reused buffers. A pool is the obvious thing to write and it is wrong:
    dispatching the tile kernel does not finish with the host array, since
    ``device_put`` is asynchronous and on the CPU backend may alias the array
    rather than copy it, so a buffer handed back after dispatch can be
    overwritten while a tile is still reading it. Owning an array per span ties
    its life to the block, and the block stays alive until its tile is written
    out. One array is a few tens of megabytes, which is far less than the
    device memory the tiles in flight already hold.
    """

    def __init__(
        self,
        loader,
        locations: Sequence,
        rotations: np.ndarray,
        shifts: np.ndarray,
        ctf: np.ndarray,
    ):
        self._loader = loader
        self._locations = list(locations)
        self._rotations = rotations
        self._shifts = shifts
        self._ctf = ctf
        descriptor = vio.query_descriptor(self._locations[0].key)
        self._image_shape = tuple(vio.get_core_extents(descriptor))

    def request(self, span: _Span) -> Tuple:
        """Start reading ``span`` into an array of its own."""
        images = np.empty((len(span), *self._image_shape), dtype=np.float32)
        completion = vio.read_batch_async(
            self._loader, images, self._locations[span.start:span.stop]
        )
        return completion, span, images

    def collect(self, request) -> _Block:
        """Wait for a request and turn it into a block."""
        completion, span, images = request
        completion.get()
        return _Block(
            span=span,
            images=images,
            rotations=self._rotations[span.start:span.stop],
            shifts=self._shifts[span.start:span.stop],
            ctf=self._ctf[span.start:span.stop],
        )


class StreamingSquaredDistanceMatrix:
    """Assemble the full pairwise common-line distance-squared matrix out of core.

    The matrix is symmetric, so only the upper triangle of tiles is computed and
    each off-diagonal tile is mirrored into its transpose. It is far too large
    to live on a GPU and often too large for host RAM, so it is built one
    ``block_size x block_size`` tile at a time.

    Everything that used to need a thread is now asynchronous on its own terms.
    vitrio reads on its worker pool and hands back a completion; ``device_put``
    and the tile kernel are asynchronous JAX calls; only ``device_get`` blocks.
    So one Python loop is enough: it keeps the next column block in flight while
    it dispatches the current tile, and drains finished tiles from a bounded
    queue of results, which is what caps how much device memory is live.

    Re-reading the block shared along a tile row costs nothing worth caching
    against when a scratch is configured, which is what the scratch is for, so
    there is no block cache here.

    The loop runs under ``jax.transfer_guard("disallow")``, so the only
    host-device copies are the explicit ``device_put`` on upload and
    ``device_get`` on download; an accidental implicit transfer raises instead.
    """

    def __init__(
        self,
        loader,
        image_locations: Sequence,
        rotations: np.ndarray,
        shifts: np.ndarray,
        defocus: np.ndarray,
        box_size: int,
        ctf_context: CtfContext,
        sigma2: np.ndarray,
        mask: Optional[np.ndarray] = None,
        *,
        low_pass_cutoff: Optional[float] = None,
        devices: Optional[Sequence[jax.Device]] = None,
        block_size: int = 64,
        pending_tiles: Optional[int] = None,
        progress: bool = True,
    ):
        self._loader = loader
        self._locations = list(image_locations)
        self._rotations = np.asarray(rotations, dtype=np.float32)
        self._shifts = np.asarray(shifts, dtype=np.float32)
        self._box_size = box_size

        self._count = len(self._locations)
        if not (len(self._rotations) == len(self._shifts) == len(defocus)
                == self._count):
            raise ValueError(
                'image_locations, rotations, shifts and defocus must describe '
                'the same number of particles'
            )

        self._devices = list(devices) if devices is not None else jax.devices()
        if not self._devices:
            raise ValueError('at least one device is required')

        # CTFs are small and cheap, so they are generated once on the CPU
        # rather than per block, which keeps the loop to reads and kernels.
        self._ctf = self._generate_ctf(np.asarray(defocus, dtype=np.float32),
                                       ctf_context)

        expected_psd_length = box_size // 2 + 1
        sigma2 = jnp.asarray(sigma2)
        if sigma2.shape != (expected_psd_length,):
            raise ValueError(
                f'sigma2 must be a 1D noise PSD profile of length '
                f'{expected_psd_length} (box_size // 2 + 1), got shape '
                f'{tuple(sigma2.shape)}'
            )
        if mask is not None and tuple(jnp.shape(mask)) != (box_size, box_size):
            raise ValueError(
                f'mask must have shape {(box_size, box_size)}, '
                f'got {tuple(jnp.shape(mask))}'
            )

        # Everything constant across tiles is parked on every device up front,
        # so the loop itself never transfers anything but the blocks.
        low_pass = (
            None if low_pass_cutoff is None
            else jnp.square(butterworth_1d(box_size, low_pass_cutoff, order=2))
        )
        self._constants = {
            device: dict(
                sigma2=jax.device_put(sigma2, device),
                mask=(None if mask is None
                      else jax.device_put(jnp.asarray(mask), device)),
                frequency_weights=(None if low_pass is None
                                   else jax.device_put(low_pass, device)),
            )
            for device in self._devices
        }

        self._block_size = block_size
        self._spans = self._build_spans()
        self._total_tiles = len(self._spans) * (len(self._spans) + 1) // 2
        # Two per device keeps every device fed while one result is draining.
        self._pending_tiles = pending_tiles or 2 * len(self._devices)
        self._progress = progress

    def run(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the ``(N, N)`` distance-squared matrix into ``out``.

        ``out`` lets the caller supply host storage, such as a ``numpy.memmap``,
        for matrices that do not fit in RAM.
        """
        shape = (self._count, self._count)
        if out is None:
            out = np.empty(shape, dtype=np.float32)
        elif out.shape != shape:
            raise ValueError(f'out must have shape {shape}, got {out.shape}')

        logger.info(
            'Computing %d x %d distance matrix as %d tiles over %d device(s)',
            self._count, self._count, self._total_tiles, len(self._devices),
        )
        blocks = _BlockLoader(
            self._loader, self._locations,
            self._rotations, self._shifts, self._ctf,
        )
        with jax.transfer_guard('disallow'), tqdm(
            total=self._total_tiles, disable=not self._progress,
            unit='tile', desc='Distance tiles',
        ) as progress:
            pending = deque()
            for index, (row, col) in enumerate(self._iter_tiles(blocks)):
                device = self._devices[index % len(self._devices)]
                # The blocks are held until the tile is written: the device
                # arrays may still refer to their host images.
                pending.append(
                    (row, col, self._compute_tile(row, col, device))
                )
                if len(pending) >= self._pending_tiles:
                    self._write(out, pending.popleft())
                    progress.update(1)
            while pending:
                self._write(out, pending.popleft())
                progress.update(1)
        return out

    def _iter_tiles(self, blocks: _BlockLoader) -> Iterator[Tuple[_Block, _Block]]:
        """Yield every upper-triangle tile with its next column already in flight."""
        spans = self._spans
        for i, row_span in enumerate(spans):
            row = blocks.collect(blocks.request(row_span))
            # The read for the next column is started before this tile is
            # handed over, so it overlaps the kernel rather than following it.
            ahead = None
            for j in range(i, len(spans)):
                col = row if j == i else blocks.collect(ahead)
                following = j + 1
                ahead = (blocks.request(spans[following])
                         if following < len(spans) else None)
                yield row, col

    def _write(self, out: np.ndarray, result) -> None:
        row_block, col_block, distances2 = result
        row, col = row_block.span, col_block.span
        # Blocks on the device, then copies to the host.
        values = jax.device_get(distances2)
        out[row.start:row.stop, col.start:col.stop] = values
        if col.index != row.index:
            out[col.start:col.stop, row.start:row.stop] = values.T

    def _compute_tile(self, row: _Block, col: _Block, device: jax.Device):
        constants = self._constants[device]
        row_device = self._upload(row, device)
        col_device = (row_device if col.span.index == row.span.index
                      else self._upload(col, device))
        return compute_distance2_tile(
            images_row=row_device[0], shifts_row=row_device[1],
            rotations_row=row_device[2], ctf_row=row_device[3],
            images_col=col_device[0], shifts_col=col_device[1],
            rotations_col=col_device[2], ctf_col=col_device[3],
            **constants,
        )

    @staticmethod
    def _upload(block: _Block, device: jax.Device):
        return tuple(
            jax.device_put(array, device)
            for array in (block.images, block.shifts,
                          block.rotations, block.ctf)
        )

    def _generate_ctf(self, defocus: np.ndarray,
                      ctf_context: CtfContext) -> np.ndarray:
        with jax.default_device(jax.devices('cpu')[0]):
            ctf = compute_ctf_1d(jnp.asarray(defocus), self._box_size,
                                 ctf_context)
        return np.asarray(ctf, dtype=np.float32)

    def _build_spans(self) -> List[_Span]:
        return [
            _Span(index, start, min(start + self._block_size, self._count))
            for index, start in enumerate(
                range(0, self._count, self._block_size)
            )
        ]
