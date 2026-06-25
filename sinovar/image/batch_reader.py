from typing import Iterable, Optional, Tuple
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
    
    # Initialize with the first loop iteration
    path = next(it)
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

    def __del__(self):
        executor = getattr(self, '_executor', None)
        if executor is not None:
            executor.shutdown(wait=False)

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
        out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Read all ``locations`` into a single contiguous array.

        If ``out`` is provided, segments are written into ``out[:total_count]``
        in submission order; the caller is responsible for sizing the buffer
        so it can hold every frame (any extra rows past the last segment are
        left untouched, which is useful for zero-padded batch buffers).
        Otherwise a fresh array is allocated and returned.
        """
        
        plans = []
        start = 0
        for filename, index_slice in _batch_files(locations):
            if index_slice is None:
                count = 1
            else:
                count = index_slice.stop - index_slice.start

            end = start + count
            plans.append((filename, index_slice, start, end))
            
            start = end
        
        particle_shape, dtype = self.deduce_batch_layout(locations)
        if out is None:
            result = np.empty((len(locations), ) + particle_shape, dtype=dtype)

        else:
            if out.shape[0] < len(locations):
                raise ValueError(
                    f'out has shape {out.shape} but storage for {len(locations)} particles are required'
                )
            if particle_shape is not None and out.shape[1:] != particle_shape:
                raise ValueError(
                    f'out particle shape {out.shape[1:]} does not match {particle_shape}'
                )
            if dtype is not None and out.dtype != dtype:
                raise ValueError(
                    f'out dtype {out.dtype} does not match segment dtype {dtype}'
                )
            result = out

        # Each worker writes to a non-overlapping slice of `result`, so the
        # concatenation is performed in parallel by the readers themselves.
        futures = [
            self._executor.submit(
                self._read_segment, filename, index_slice, result[start:end]
            )
            for filename, index_slice, start, end in plans
        ]
        for f in futures:
            f.result()

        return result

    def _read_segment(
        self,
        filename: str,
        index_slice: Optional[slice],
        out: np.ndarray,
    ) -> None:
        mrc = self._read_file(filename)
        data = mrc.data

        if data.ndim == 4:
            if data.shape[1] != 1:
                raise RuntimeError('Expected image stack')
            data = data[:,0,:,:]

        if mrc.is_image_stack() or mrc.is_volume_stack():
            if index_slice is None:
                raise RuntimeError(
                    'Image index should be provided for image stacks'
                )
            np.copyto(out, data[index_slice])
        else:
            np.copyto(out, data[None])

    def _read_file(self, filename: str):
        with self._cache_lock:
            mrc = self._open_files.get(filename, None)

            if mrc is None:
                mrc = mrcfile.mmap(self._make_abs_filename(filename), 'r')
                self._open_files[filename] = mrc
                if len(self._open_files) >= self._max_open:
                    self._open_files.popitem(last=False)
            else:
                self._open_files.move_to_end(filename, last=True)

            assert mrc is not None
            return mrc
    
    def _make_abs_filename(self, filename: str) -> str:
        if self._prefix is not None:
            filename = os.path.join(self._prefix, filename)
            
        return filename
    