"""Reading particle images, on top of vitrio.

vitrio owns the file formats, the reader cache, the worker pool and the
optional scratch, so this module is only the thin layer that turns sinovar's
own vocabulary into vitrio's: star-file image names into locations, and a
scratch written on a command line into a scratch object.
"""
from typing import Iterable, Optional, Sequence
import os

import numpy as np
import vitrio as vio

__all__ = [
    'ImageLocation',
    'batch_destination',
    'make_locations',
    'open_scratch',
    'read_batch',
    'SCRATCH_HELP',
]

ImageLocation = vio.ImageLocation

SCRATCH_HELP = (
    "Where to keep a second copy of the particles, which is worth it when "
    "they are read many times, as the tiled distance matrix does. 'none' "
    "reads the files every time. 'memory' keeps the copy in RAM. Any other "
    "value is a path: an existing directory holds one file per data set and "
    "is shared between runs, anything else is a single file. Put that path on "
    "fast local storage when the particles live on a slow or remote disk."
)


def make_locations(
    image_names: Iterable[str],
    prefix: Optional[str] = None,
) -> list:
    """Turn ``index@file`` names from a star file into vitrio locations.

    The prefix is folded in here so that nothing downstream has to carry it:
    a location is self-contained from this point on.
    """
    locations = []
    for name in image_names:
        index, _, path = str(name).partition('@')
        if not path:
            index, path = None, str(name)
        if prefix:
            path = os.path.join(prefix, path)
        locations.append(
            vio.ImageLocation(path) if index is None
            # Star files count images from one, vitrio from zero.
            else vio.ImageLocation(path, int(index) - 1)
        )
    return locations


def batch_destination(
    locations: Sequence,
    count: Optional[int] = None,
    dtype=np.float32,
) -> np.ndarray:
    """Allocate the host array a batch of ``locations`` reads into."""
    if not locations:
        raise ValueError('at least one location is required')
    descriptor = vio.query_descriptor(locations[0].key)
    shape = (len(locations) if count is None else count,
             *vio.get_core_extents(descriptor))
    return np.empty(shape, dtype=dtype)


def read_batch(loader, locations: Sequence, dtype=np.float32) -> np.ndarray:
    """Read one image per location and wait for them all."""
    destination = batch_destination(locations, dtype=dtype)
    vio.read_batch_async(loader, destination, locations).get()
    return destination


def open_scratch(
    spec: Optional[str],
    locations: Sequence,
    run_length: int,
    budget: Optional[int] = None,
):
    """Build the scratch named by ``spec``, or ``None`` for no scratch.

    ``spec`` is 'none', 'memory', or a path. A path that is an existing
    directory gets a directory scratch, which several runs can share and which
    vitrio keeps within ``budget``; any other path gets a single mapped file.
    """
    if spec in (None, '', 'none'):
        return None
    if spec == 'memory':
        return vio.create_host_image_scratch(
            locations, run_length=run_length, max_size=budget,
        )
    if os.path.isdir(spec):
        if budget is None:
            raise ValueError(
                'a directory scratch needs a budget: pass --scratch-budget'
            )
        return vio.create_directory_image_scratch(
            locations, directory=spec, run_length=run_length, budget=budget,
        )
    return vio.create_mapped_file_image_scratch(
        locations, path=spec, run_length=run_length, max_size=budget,
    )
