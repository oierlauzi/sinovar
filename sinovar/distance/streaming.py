from typing import Iterator, Optional, Sequence, Tuple
from dataclasses import dataclass
from collections import OrderedDict
from threading import Event, Lock, Thread
import logging
import queue

import numpy as np
import jax
from tqdm import tqdm

from .tile import compute_distance2_tile
from ..image import BatchReader, ImageLocation
from ..ctf import CtfContext, compute_ctf_1d

logger = logging.getLogger(__name__)

# Pushed through the pipeline queues to mark "no more work past this point".
_SENTINEL = object()
# How often blocked queue operations wake up to notice an abort, in seconds.
_POLL_SECONDS = 0.5


class _Aborted(Exception):
    """Raised inside a worker once another stage has failed, to unwind it."""


@dataclass
class _Block:
    """A contiguous slab of particles held in host memory."""

    index: int
    start: int
    stop: int
    images: np.ndarray      # (n, H, W) float32
    rotations: np.ndarray   # (n, 3, 3) float32
    shifts: np.ndarray      # (n, 2)    float32
    ctf: np.ndarray         # (n, F)    float32


@dataclass
class _DeviceBlock:
    """A block uploaded to a single device, ready to feed the tile kernel."""

    images: jax.Array
    rotations: jax.Array
    shifts: jax.Array
    ctf: jax.Array

    @classmethod
    def upload(cls, block: _Block, device: jax.Device) -> "_DeviceBlock":
        return cls(
            images=jax.device_put(block.images, device),
            rotations=jax.device_put(block.rotations, device),
            shifts=jax.device_put(block.shifts, device),
            ctf=jax.device_put(block.ctf, device),
        )


@dataclass
class _LoadedTile:
    """A pair of host-resident blocks whose distances still need computing."""

    row: _Block
    col: _Block


@dataclass
class _ComputedTile:
    """A dispatched tile: blocks identify where it lands, the array is a future."""

    row: _Block
    col: _Block
    distances2: jax.Array   # (n_row, n_col), still resident on its device


class _BlockCache:
    """Single-reader LRU cache of recently loaded blocks.

    Row-major tile iteration keeps one row block hot for the whole length of a
    row while the column block cycles, so even a tiny cache eliminates the bulk
    of the redundant disk reads without ever holding the full dataset in memory.
    Cached and in-flight blocks share the same arrays by reference, so caching
    costs nothing beyond the blocks it deliberately retains.
    """

    def __init__(self, capacity: int, build_block):
        self._capacity = max(capacity, 2)
        self._build_block = build_block
        self._blocks: "OrderedDict[int, _Block]" = OrderedDict()

    def get(self, index: int) -> _Block:
        block = self._blocks.get(index)
        if block is None:
            block = self._build_block(index)
            self._blocks[index] = block
            if len(self._blocks) > self._capacity:
                self._blocks.popitem(last=False)
        else:
            self._blocks.move_to_end(index)
        return block


class StreamingSquaredDistanceMatrix:
    """Assemble the full pairwise common-line distance-squared matrix out of core.

    The matrix is treated as symmetric --- only the upper triangle of tiles is
    computed and each off-diagonal tile is mirrored into its transpose --- and it
    is far too large to live on a GPU, often even in host RAM. It is therefore
    built one ``block_size x block_size`` tile at a time and streamed through a
    three-stage software pipeline:

    1. **load** (one background thread) reads each tile's two image blocks from
       disk into host memory and generates their CTFs on the CPU, reusing an LRU
       :class:`_BlockCache` so the block shared along a row is neither re-read nor
       re-generated for every column.
    2. **upload + compute** (the calling thread) moves a loaded tile onto a
       device and dispatches :func:`compute_distance2_tile`. Both are
       asynchronous JAX operations that return immediately, so this thread races
       ahead issuing work and round-robins tiles across every device. Upload and
       compute are not split into separate threads precisely because the device
       transfer is already asynchronous --- a dedicated upload thread would only
       add a hand-off, not overlap.
    3. **download + write** (a pool of background threads) blocks on each device
       result, copies it to the host and writes it --- and its transpose --- into
       the output matrix.

    Bounded queues between the stages provide back-pressure, capping how much
    image data and how many in-flight device computations are alive at once.
    Distinct tiles map to disjoint regions of the output, so the writers never
    contend and need no lock.

    Every stage runs under ``jax.transfer_guard("disallow")``, so the only
    host<->device copies are the explicit ``device_put`` on upload and
    ``device_get`` on download; any accidental implicit transfer raises instead.
    """

    def __init__(
        self,
        image_reader: BatchReader,
        image_locations: Sequence[ImageLocation],
        rotations: np.ndarray,
        shifts: np.ndarray,
        defocus: np.ndarray,
        box_size: int,
        ctf_context: CtfContext,
        *,
        devices: Optional[Sequence[jax.Device]] = None,
        block_size: int = 64,
        prefetch_tiles: int = 4,
        cache_blocks: int = 4,
        writeback_workers: Optional[int] = None,
        progress: bool = True,
    ):
        self._image_reader = image_reader
        self._image_locations = list(image_locations)
        self._rotations = np.asarray(rotations, dtype=np.float32)
        self._shifts = np.asarray(shifts, dtype=np.float32)
        self._defocus = np.asarray(defocus, dtype=np.float32)
        self._box_size = box_size
        self._ctf_context = ctf_context
        # CTFs are cheap, so they are generated alongside the disk read and kept
        # on the CPU where they never compete with the distance kernels for a GPU.
        self._ctf_device = jax.devices("cpu")[0]

        self._count = len(self._image_locations)
        if not (len(self._rotations) == len(self._shifts) == len(self._defocus) == self._count):
            raise ValueError(
                "image_locations, rotations, shifts and defocus must describe "
                "the same number of particles"
            )

        self._devices = list(devices) if devices is not None else jax.devices()
        if not self._devices:
            raise ValueError("at least one device is required")

        self._block_size = block_size
        self._block_count = -(-self._count // block_size)  # ceil division
        self._total_tiles = self._block_count * (self._block_count + 1) // 2

        self._cache = _BlockCache(cache_blocks, self._build_block)
        self._load_queue: "queue.Queue" = queue.Queue(maxsize=prefetch_tiles)
        self._result_queue: "queue.Queue" = queue.Queue(maxsize=2 * len(self._devices))
        self._writeback_workers = writeback_workers or len(self._devices)
        self._progress = progress

        self._abort = Event()
        self._error: Optional[BaseException] = None
        self._progress_lock = Lock()
        self._progress_bar: Optional[tqdm] = None
        self._out: Optional[np.ndarray] = None

    def compute(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the ``(N, N)`` distance-squared matrix into ``out`` (or a fresh array).

        ``out`` lets the caller supply host storage --- e.g. a ``numpy.memmap`` ---
        for matrices that do not fit in RAM.
        """
        shape = (self._count, self._count)
        if out is None:
            out = np.empty(shape, dtype=np.float32)
        elif out.shape != shape:
            raise ValueError(f"out must have shape {shape}, got {out.shape}")
        self._out = out

        logger.info(
            "Computing %d x %d distance matrix as %d tiles over %d device(s)",
            self._count, self._count, self._total_tiles, len(self._devices),
        )

        loader = Thread(target=self._run_guarded, args=(self._load_loop,), name="distance-loader")
        writers = [
            Thread(target=self._run_guarded, args=(self._writeback_loop,), name=f"distance-writeback-{k}")
            for k in range(self._writeback_workers)
        ]
        self._progress_bar = tqdm(
            total=self._total_tiles,
            disable=not self._progress,
            unit="tile",
            desc="Distance tiles",
        )
        loader.start()
        for writer in writers:
            writer.start()

        try:
            self._run_guarded(self._dispatch_loop)
        except _Aborted:
            pass
        finally:
            loader.join()
            for writer in writers:
                writer.join()
            self._progress_bar.close()

        if self._error is not None:
            raise self._error
        return out

    def _run_guarded(self, loop) -> None:
        # The transfer guard is thread-local, so every worker re-establishes it.
        # Implicit host<->device copies then raise, leaving the explicit
        # device_put (on upload) and device_get (on download) as the only
        # transfers in the pipeline.
        with jax.transfer_guard("disallow"):
            loop()

    def _load_loop(self) -> None:
        try:
            for i, j in self._iter_tiles():
                row = self._cache.get(i)
                col = row if j == i else self._cache.get(j)
                self._put(self._load_queue, _LoadedTile(row, col))
            self._put(self._load_queue, _SENTINEL)
        except _Aborted:
            pass
        except BaseException as error:  # noqa: BLE001 - surfaced to the caller
            self._fail(error)

    def _dispatch_loop(self) -> None:
        tile_index = 0
        while True:
            tile = self._get(self._load_queue)
            if tile is _SENTINEL:
                break
            device = self._devices[tile_index % len(self._devices)]
            distances2 = self._compute_tile(tile, device)
            self._put(self._result_queue, _ComputedTile(tile.row, tile.col, distances2))
            tile_index += 1

        for _ in range(self._writeback_workers):
            self._put(self._result_queue, _SENTINEL)

    def _writeback_loop(self) -> None:
        try:
            while True:
                tile = self._get(self._result_queue)
                if tile is _SENTINEL:
                    return

                distances2 = jax.device_get(tile.distances2)  # blocks on the device, copies to host
                row, col = tile.row, tile.col
                self._out[row.start:row.stop, col.start:col.stop] = distances2
                if col.index != row.index:
                    self._out[col.start:col.stop, row.start:row.stop] = distances2.T

                self._record_progress()
        except _Aborted:
            pass
        except BaseException as error:  # noqa: BLE001 - surfaced to the caller
            self._fail(error)

    def _compute_tile(self, tile: _LoadedTile, device: jax.Device) -> jax.Array:
        row = _DeviceBlock.upload(tile.row, device)
        col = row if tile.col.index == tile.row.index else _DeviceBlock.upload(tile.col, device)
        return compute_distance2_tile(
            images_row=row.images,
            shifts_row=row.shifts,
            rotations_row=row.rotations,
            ctf_row=row.ctf,
            images_col=col.images,
            shifts_col=col.shifts,
            rotations_col=col.rotations,
            ctf_col=col.ctf,
        )

    def _build_block(self, index: int) -> _Block:
        start = index * self._block_size
        stop = min(start + self._block_size, self._count)
        images = np.asarray(
            self._image_reader.read_batch(self._image_locations[start:stop]),
            dtype=np.float32,
        )
        return _Block(
            index=index,
            start=start,
            stop=stop,
            images=images,
            rotations=self._rotations[start:stop],
            shifts=self._shifts[start:stop],
            ctf=self._generate_ctf(start, stop),
        )

    def _generate_ctf(self, start: int, stop: int) -> np.ndarray:
        defocus = jax.device_put(self._defocus[start:stop], self._ctf_device)
        ctf = compute_ctf_1d(defocus, self._box_size, self._ctf_context)
        return np.asarray(jax.device_get(ctf), dtype=np.float32)

    def _iter_tiles(self) -> Iterator[Tuple[int, int]]:
        for i in range(self._block_count):
            for j in range(i, self._block_count):
                yield i, j

    def _record_progress(self) -> None:
        # Several writeback threads share one bar, so the update is serialized.
        with self._progress_lock:
            self._progress_bar.update(1)

    def _fail(self, error: BaseException) -> None:
        with self._progress_lock:
            if self._error is None:
                self._error = error
        self._abort.set()

    def _put(self, q: "queue.Queue", item) -> None:
        while not self._abort.is_set():
            try:
                q.put(item, timeout=_POLL_SECONDS)
                return
            except queue.Full:
                continue
        raise _Aborted()

    def _get(self, q: "queue.Queue"):
        while not self._abort.is_set():
            try:
                return q.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                continue
        raise _Aborted()
