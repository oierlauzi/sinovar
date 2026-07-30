"""Reading a basis of volumes off disk.

RECOVAR (and every comparable method) writes its principal volumes as one MRC
file each, so the natural input is a list of paths. Three ways of giving that
list are accepted, in decreasing order of how much a shell can do for you:

* several ``.mrc`` paths, which lets the shell glob them
  (``eigen_volume*.mrc``) --- note that a glob must sort into basis order, which
  is what zero-padded names do;
* one ``.mrc`` file holding a *volume stack*, i.e. the whole basis in one file;
* one ``.txt`` file with a path per line (``#`` comments and blank lines are
  ignored). Relative paths inside it are resolved against the list's own
  directory when they do not resolve as given, so a list travels with the
  volumes it names.

The order of the list *is* the order of the basis: volume ``k`` must match
component ``k`` of the embedding, since the model reads a pair of particles as
``V_i - V_j = U (z_i - z_j)``.

The volumes are read one file at a time rather than as a basis, because a basis
at its original box size is far larger than anything downstream keeps: a hundred
eigen-volumes of box 256 are 6.7 GB, and the caller resamples each one before
holding on to it.
"""
from typing import List, Optional, Sequence, Tuple
import logging
import os

import mrcfile
import numpy as np

logger = logging.getLogger(__name__)

#: Extension read as a list of volumes rather than as a volume.
_LIST_SUFFIX = '.txt'


def _read_list_file(path: str) -> List[str]:
    """Read the volume paths named by a ``.txt`` list, one per line."""
    with open(path) as lines:
        entries = [
            stripped for stripped in (line.strip() for line in lines)
            if stripped and not stripped.startswith('#')
        ]
    if not entries:
        raise ValueError(f'{path} names no volumes')

    # A list is usually kept next to the volumes it names, so fall back to
    # resolving against its directory rather than the working directory.
    directory = os.path.dirname(os.path.abspath(path))
    return [
        entry if os.path.exists(entry) else os.path.join(directory, entry)
        for entry in entries
    ]


def expand_volume_paths(paths: Sequence[str]) -> List[str]:
    """Resolve a volume-basis argument into a list of MRC paths.

    A single ``.txt`` list is expanded; anything else is returned as given. A
    single MRC holding a volume stack stays one path --- it is
    :func:`read_volumes` that unpacks its slices.
    """
    if not paths:
        raise ValueError('no volumes given')

    if len(paths) == 1 and paths[0].endswith(_LIST_SUFFIX):
        expanded = _read_list_file(paths[0])
        logger.info('Expanded %s into %d volume path(s)', paths[0], len(expanded))
        return expanded

    for path in paths:
        if path.endswith(_LIST_SUFFIX):
            raise ValueError(
                f'{path} is a list of volumes, so it must be the only argument'
            )
    return list(paths)


def read_volume_box_size(path: str) -> int:
    """Box size of the volume(s) in one MRC, from its header alone.

    Choosing the grid a refinement works on needs the box size before any volume
    is worth reading, which is what this is for.
    """
    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        box_size = int(mrc.header.nx)
        if int(mrc.header.ny) != box_size:
            raise ValueError(
                f'{path} is not cubic: {box_size} x {int(mrc.header.ny)}'
            )
    return box_size


def read_volumes(path: str) -> Tuple[np.ndarray, Optional[float]]:
    """Read one MRC as ``(K, B, B, B)``, plus its voxel size if recorded.

    ``K`` is 1 for a plain volume and the stack depth for a volume stack.
    """
    with mrcfile.open(path, permissive=True) as mrc:
        volumes = np.asarray(mrc.data, dtype=np.float32)
        voxel_size = float(mrc.voxel_size.x) or None

    if volumes.ndim == 3:
        volumes = volumes[None]
    elif volumes.ndim != 4:
        raise ValueError(f'{path} is neither a volume nor a volume stack')

    box_size = volumes.shape[-1]
    if any(dimension != box_size for dimension in volumes.shape[1:]):
        raise ValueError(f'{path} is not cubic: {volumes.shape[1:]}')

    return volumes, voxel_size
