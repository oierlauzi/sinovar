from typing import Iterable, List, Optional, Tuple
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
import mrcfile
import os

from .image_location import ImageLocation

def _index_or_none(position_in_stack: Optional[int]) -> Optional[int]:
    return None if position_in_stack is None else position_in_stack - 1

def _batch_files(paths: Iterable[ImageLocation]):
    it = iter(paths)

    # Initialize with the first loop iteration. An empty input yields nothing.
    try:
        path = next(it)
    except StopIteration:
        return
    current_filename = path.filename
    current_end = path.position_in_stack
    current_start = _index_or_none(current_end)

    for path in it:
        filename = path.filename
        index = _index_or_none(path.position_in_stack)

        if filename == current_filename and index == current_end and current_end is not None:
            current_end += 1

        else:
            if current_start is not None:
                assert (current_end is not None)
                yield current_filename, slice(current_start, current_end)
            else:
                yield current_filename, None

            current_filename = path.filename
            current_end = path.position_in_stack
            current_start = _index_or_none(current_end)

    if current_start is not None:
        assert (current_end is not None)
        yield current_filename, slice(current_start, current_end)
    else:
        yield current_filename, None


# A single coalesced run inside one file: ``index_slice`` selects the source
# rows (``None`` for a non-stack single image) and ``[start, stop)`` is the
# destination range in the output batch.
_Segment = Tuple[Optional[slice], int, int]


class BatchReader:
    def __init__(
        self,
        prefix: Optional[str] = None,
        max_open: int = 25,
        max_workers: int = 8,
    ):
        self._open_files = OrderedDict()
        self._prefix = prefix
        self._max_open = max_open
        self._cache_lock = Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='BatchReader',
        )

    def __enter__(self) -> 'BatchReader':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        self.close()

    def close(self) -> None:
        """Release the worker pool and every cached file handle.

        Only call this once no reads are in flight: the worker threads hold live
        references to the files they are reading, so the open-file cache is only
        emptied (closing the underlying mmaps) once they are quiescent.
        """
        executor = getattr(self, '_executor', None)
        if executor is not None:
            executor.shutdown(wait=False)

        open_files = getattr(self, '_open_files', None)
        lock = getattr(self, '_cache_lock', None)
        if open_files is None or lock is None:
            return
        with lock:
            files = list(open_files.values())
            open_files.clear()
        for mrc in files:
            mrc.close()

    def deduce_batch_layout(
        self,
        locations: Iterable[ImageLocation]
    ) -> Tuple[Tuple[int, ...], np.dtype]:
        ite = iter(locations)

        try:
            item = next(ite)
        except StopIteration:
            raise ValueError('Can not deduce batch layout from no locations')

        mrc = self._read_file(item.filename)
        data = mrc.data
        return data.shape[-2:], data.dtype

    def read_batch(
        self,
        locations: Iterable[ImageLocation],
        out: Optional[np.ndarray] = None,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """Read all ``locations`` into a single contiguous array.

        ``locations`` is consumed once, so any iterable (including a one-shot
        iterator) is accepted. Contiguous runs within a file are coalesced into a
        single slice read, and every run targeting the same file is served by a
        single worker, so the disk reads run in parallel across files.

        ``dtype`` selects the output element type; when given (e.g. ``float32``),
        the cast from the native MRC dtype is fused into the read, avoiding a
        second full-batch allocation and copy in the caller. It defaults to the
        file's native dtype.

        If ``out`` is provided, segments are written into ``out[:total_count]``
        in submission order; the caller is responsible for sizing the buffer
        so it can hold every frame (any extra rows past the last segment are
        left untouched, which is useful for zero-padded batch buffers).
        Otherwise a fresh array is allocated and returned.
        """

        # Single pass over ``locations``: coalesce into runs and group the runs
        # by file so each file is opened and read by exactly one worker.
        segments_by_file: "OrderedDict[str, List[_Segment]]" = OrderedDict()
        total = 0
        first_filename: Optional[str] = None
        for filename, index_slice in _batch_files(locations):
            if first_filename is None:
                first_filename = filename
            count = 1 if index_slice is None else index_slice.stop - index_slice.start
            start = total
            total = start + count
            segments_by_file.setdefault(filename, []).append((index_slice, start, total))

        if total == 0:
            raise ValueError('Can not read a batch from no locations')

        # The first file is opened here and reused (cached) by its worker below,
        # so the layout probe costs no extra open.
        first = self._read_file(first_filename)
        particle_shape = tuple(first.data.shape[-2:])
        target_dtype = first.data.dtype if dtype is None else np.dtype(dtype)

        if out is None:
            result = np.empty((total,) + particle_shape, dtype=target_dtype)
        else:
            if out.shape[0] < total:
                raise ValueError(
                    f'out has shape {out.shape} but storage for {total} particles are required'
                )
            if out.shape[1:] != particle_shape:
                raise ValueError(
                    f'out particle shape {out.shape[1:]} does not match {particle_shape}'
                )
            if out.dtype != target_dtype:
                raise ValueError(
                    f'out dtype {out.dtype} does not match target dtype {target_dtype}'
                )
            result = out

        # Each worker writes to disjoint rows of `result`, so the concatenation
        # is performed in parallel by the readers themselves with no locking.
        futures = [
            self._executor.submit(self._read_file_into, filename, segments, result)
            for filename, segments in segments_by_file.items()
        ]
        for f in futures:
            f.result()

        return result

    def _read_file_into(
        self,
        filename: str,
        segments: List[_Segment],
        result: np.ndarray,
    ) -> None:
        mrc = self._read_file(filename)
        data = mrc.data

        if data.ndim == 4:
            if data.shape[1] != 1:
                raise RuntimeError('Expected image stack')
            data = data[:, 0, :, :]

        if not (mrc.is_image_stack() or mrc.is_volume_stack()):
            # Single image: each requested run is one copy of it. Assignment
            # casts to ``result``'s dtype.
            for _, start, stop in segments:
                np.copyto(result[start:stop], data[None])
            return

        if len(segments) == 1:
            # The common case (a contiguous run, e.g. a whole stack block) is a
            # single slice copy --- no index arrays to build.
            index_slice, start, stop = segments[0]
            if index_slice is None:
                raise RuntimeError('Image index should be provided for image stacks')
            np.copyto(result[start:stop], data[index_slice])
            return

        # Several disjoint runs from the same stack (scattered indices): gather
        # every needed row in one indexed read and scatter it into place, casting
        # to ``result``'s dtype. This collapses what would otherwise be many tiny
        # per-run tasks into a single optimized copy.
        src: List[int] = []
        dst: List[int] = []
        for index_slice, start, stop in segments:
            if index_slice is None:
                raise RuntimeError('Image index should be provided for image stacks')
            src.extend(range(index_slice.start, index_slice.stop))
            dst.extend(range(start, stop))
        src_idx = np.fromiter(src, dtype=np.intp, count=len(src))
        dst_idx = np.fromiter(dst, dtype=np.intp, count=len(dst))
        result[dst_idx] = data[src_idx]

    def _read_file(self, filename: str):
        with self._cache_lock:
            mrc = self._open_files.get(filename, None)
            if mrc is not None:
                self._open_files.move_to_end(filename, last=True)
                return mrc

        # Open outside the cache lock so opens of distinct files proceed
        # concurrently instead of serializing on a single mmap+header parse.
        opened = mrcfile.mmap(self._make_abs_filename(filename), 'r')

        with self._cache_lock:
            existing = self._open_files.get(filename, None)
            if existing is not None:
                # Another worker opened the same file while we were; keep theirs.
                self._open_files.move_to_end(filename, last=True)
                opened.close()
                return existing

            self._open_files[filename] = opened
            # Evict by dropping the dict reference only: a worker may still hold
            # and be reading an evicted file, so its mmap is closed by refcount
            # once that worker releases it --- never out from under a live read.
            while len(self._open_files) > self._max_open:
                self._open_files.popitem(last=False)
            return opened

    def _make_abs_filename(self, filename: str) -> str:
        if self._prefix is not None:
            filename = os.path.join(self._prefix, filename)

        return filename
