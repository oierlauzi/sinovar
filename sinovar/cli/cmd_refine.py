from typing import Optional, Sequence
import argparse
import logging

import jax
import jax.numpy as jnp
import numpy as np
import starfile
from tqdm import tqdm

from .. import ctf
from .. import image
from .. import mask
from .. import noise
from .. import refine
from .. import resample
from ..annotate import EMBEDDING_COLUMN, parse_embedding_column
from ..filter import butterworth_1d
from .common import Particles, add_device_argument, read_particles, select_devices

logger = logging.getLogger(__name__)

COMMAND = 'refine'
HELP = 'Refine an existing embedding against the common-line distances it predicts'

#: How far past the requested resolution the working box keeps frequencies. The
#: low-pass filter is a soft Butterworth, so cutting the band exactly at its
#: cut-off would truncate a tail the images still carry and the model then could
#: not; a factor above one keeps that tail inside the band.
_BAND_OVERSAMPLING = 1.5

#: Rolloff of the particle mask at the full box size, in pixels; scaled with the
#: working box so that the masked area stays the same physical disc.
_MASK_ROLLOFF = 16.0

#: Smallest box the working grid is chosen to be, whatever the resolution asks
#: for: below this the mask and the CTF have too few samples to mean anything.
_MINIMUM_BOX = 16

#: Particles read from disk per resampling batch.
_LOAD_BLOCK = 1024


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='STAR',
        help='Input STAR file whose particles already carry a '
             f'`{EMBEDDING_COLUMN}` column, e.g. a RECOVAR embedding'
    )
    parser.add_argument(
        '-u', '--volumes',
        required=True,
        nargs='+',
        metavar='MRC',
        help='Eigen-volumes spanning the embedding, one MRC each and in the '
             'order of the embedding components (a glob works if the names sort '
             'that way). A single MRC holding a volume stack, or a single .txt '
             'file with one path per line, are also accepted'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        metavar='STAR',
        help='Output STAR file with the refined embedding'
    )
    parser.add_argument(
        '--prefix',
        metavar='DIR',
        help='Prefix for the MRC binary files.'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=4.0,
        help='Maximum resolution in angstrom'
    )
    parser.add_argument(
        '--diameter',
        type=float,
        required=True,
        help='Particle diameter in angstrom'
    )
    parser.add_argument(
        '--volume_box',
        type=int,
        default=None,
        help='Box size everything is resampled to before refining. Defaults to '
             'the smallest box that resolves --resolution with room for the '
             'low-pass tail, which is what makes the projections affordable; '
             'raise it to model finer detail, at a cost cubic in the box'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Number of passes over the dataset; each updates every particle '
             'once, so this is both the iteration count and the cost'
    )
    parser.add_argument(
        '--anchors',
        type=int,
        default=8,
        help='Particles updated per step. Each costs one projection of the '
             'whole basis, which is the bulk of a step'
    )
    parser.add_argument(
        '--partners',
        type=int,
        default=256,
        help='Random pairs drawn for each anchor, which is how much information '
             'one update has; it needs to exceed the number of components '
             'comfortably. Cheaper than an anchor, up to roughly the box size'
    )
    parser.add_argument(
        '--ridge',
        type=float,
        default=1e-2,
        help='Damping of the per-anchor solve, relative to the mean curvature '
             'of its pairs; it is what carries the directions those pairs '
             'constrain weakly'
    )
    parser.add_argument(
        '--huber',
        type=float,
        default=1.0,
        help='Residual at which a pair stops pulling harder, in units of the '
             'RMS measured distance; bounds the influence of a badly measured '
             'common line'
    )
    parser.add_argument(
        '--step_size',
        type=float,
        default=1.0,
        help='Relaxation of the majorization step, where 1 minimizes the '
             'majorization exactly; below that damps it'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=None,
        help='Global factor from volume units to image units. Fitted from the '
             'sampled pairs by default, which pins the refined embedding to the '
             'units the input arrived in; give it explicitly to refine those too'
    )
    parser.add_argument(
        '--floor_scale',
        type=float,
        default=None,
        help='Factor on the calibrated noise variance. Fitted from the data by '
             'default and reported in the log; it answers for every error that is '
             'independent between the two images of a pair, so a value far from 1 '
             'measures unmodelled error rather than only a wrong sigma2. Give it '
             'explicitly (1 to trust the estimate) when that fit overreaches'
    )
    parser.add_argument(
        '--no_noise_floor',
        dest='noise_floor',
        action='store_false',
        help='Fit the raw measured distances instead of modelling the noise '
             'they carry. The noise inflates every measurement, so this biases '
             'the embedding outwards'
    )
    parser.add_argument(
        '--partner_batch',
        type=int,
        default=8,
        help='Partners whose common lines are extracted together; trades device '
             'memory against parallelism'
    )
    parser.add_argument(
        '--anchor_batch',
        type=int,
        default=1,
        help='Anchors projected together; trades device memory against '
             'parallelism'
    )
    parser.add_argument(
        '--slab',
        type=int,
        default=8,
        help='Integration steps resampled at once when projecting the basis; '
             'trades device memory against parallelism'
    )
    add_device_argument(parser)


def _select_working_box(
    box_size: int,
    volume_box: int,
    pixel_size: float,
    resolution: float,
    requested: Optional[int]
) -> int:
    """Pick the box size the images and the volumes are compared at.

    Projecting the basis costs the box size cubed and a common line the box size
    squared, while the objective only looks at frequencies below
    ``--resolution``: resampling down to the smallest box that holds them is
    therefore the difference between a tractable refinement and an impossible
    one. Neither the images nor the volumes can be asked for more than they
    have, so the box never grows past either.
    """
    limit = min(box_size, volume_box)
    if requested is not None:
        if requested > limit:
            raise ValueError(
                f'--volume_box {requested} exceeds what the data holds: the '
                f'images are {box_size} and the eigen-volumes {volume_box}'
            )
        return requested

    # A box of B resolves B / 2 frequency bins, and the band wanted is
    # oversampling * (pixel_size / resolution) * box_size of them.
    cutoff = pixel_size / resolution
    needed = 2.0 * _BAND_OVERSAMPLING * cutoff * box_size
    # Even, and never so small that the mask has no disc left to speak of.
    return int(np.clip(2 * int(np.ceil(0.5 * needed)), min(_MINIMUM_BOX, limit), limit))


def _load_working_images(
    reader: image.BatchReader,
    particles: Particles,
    working_box: int
) -> np.ndarray:
    """Read every particle and resample it to the working box, into host memory.

    The refinement visits pairs at random, so the images have to be reachable
    without hitting the disk for each one; band-limited they are small enough
    that the whole dataset fits in host memory, which is the other half of what
    the resampling buys. Only the batch a step needs is ever uploaded.
    """
    count = len(particles)
    images = np.empty((count, working_box, working_box), dtype=np.float32)
    logger.info(
        'Reading %d particle(s) resampled to box %d (%.1f GB in host memory)',
        count, working_box, images.nbytes / 1e9,
    )

    for start in tqdm(
        range(0, count, _LOAD_BLOCK), unit='block', desc='Particle blocks'
    ):
        stop = min(start + _LOAD_BLOCK, count)
        batch = reader.read_batch(
            particles.locations.iloc[start:stop], dtype=np.float32
        )
        if batch.shape[-1] != particles.box_size:
            raise ValueError(
                f'the images are {batch.shape[-1]} pixels wide but the optics '
                f'table says {particles.box_size}'
            )
        images[start:stop] = np.asarray(
            resample.fourier_crop_2d(jnp.asarray(batch), working_box)
        )

    return images


def _load_volume_basis(
    paths: Sequence[str],
    particles: Particles,
    volume_box: int,
    n_components: int,
    working_box: int
) -> np.ndarray:
    """Read the first ``n_components`` eigen-volumes onto the working grid.

    Resampled as they are read, one file at a time: a basis at its original box
    size is an order of magnitude larger than the one the refinement works with,
    and only the latter needs to exist.
    """
    basis = []
    voxel_size = None
    for path in paths:
        if len(basis) >= n_components:
            break

        volumes, own_voxel_size = image.read_volumes(path)
        if voxel_size is None:
            voxel_size = own_voxel_size
        if volumes.shape[-1] != volume_box:
            raise ValueError(
                f'{path} has box size {volumes.shape[-1]}, but the basis started '
                f'at {volume_box}'
            )

        for volume in volumes[:n_components - len(basis)]:
            basis.append(np.asarray(
                resample.fourier_crop_3d(jnp.asarray(volume), working_box)
            ))

    if len(basis) < n_components:
        raise ValueError(
            f'the embedding has {n_components} component(s) but only '
            f'{len(basis)} eigen-volume(s) were given'
        )
    logger.info('Read %d eigen-volume(s), one per embedding component', len(basis))

    if voxel_size is not None:
        extent = voxel_size * volume_box
        expected = particles.pixel_size * particles.box_size
        if abs(extent / expected - 1.0) > 0.01:
            logger.warning(
                'The eigen-volumes span %.4g A but the particles span %.4g A. '
                'They must cover the same field of view, so check the voxel '
                'size in the volume headers.', extent, expected,
            )

    return np.stack(basis)


def run(args: argparse.Namespace) -> None:
    devices = select_devices(args.device)
    if len(devices) > 1:
        logger.warning(
            'The refinement runs on a single device; using %s', devices[0]
        )

    with jax.default_device(devices[0]):
        _run(args)


def _run(args: argparse.Namespace) -> None:
    logger.info('Reading input')
    particles = read_particles(args.input)
    if EMBEDDING_COLUMN not in particles.metadata:
        raise ValueError(
            f'{args.input} has no {EMBEDDING_COLUMN} column; run `sinovar embed` '
            f'first, or import the embedding to refine'
        )
    embedding = parse_embedding_column(particles.metadata[EMBEDDING_COLUMN])
    n_components = embedding.shape[1]

    volume_paths = image.expand_volume_paths(args.volumes)
    volume_box = image.read_volume_box_size(volume_paths[0])
    working_box = _select_working_box(
        particles.box_size,
        volume_box,
        particles.pixel_size,
        args.resolution,
        args.volume_box,
    )
    working_pixel_size = particles.pixel_size * particles.box_size / working_box
    logger.info(
        'Refining at box %d (%.4g A per pixel, %.4g A Nyquist)',
        working_box, working_pixel_size, 2.0 * working_pixel_size,
    )

    volumes = _load_volume_basis(
        volume_paths, particles, volume_box, n_components, working_box
    )
    image_reader = image.BatchReader(prefix=args.prefix)
    images = _load_working_images(image_reader, particles, working_box)

    logger.info('Estimating noise spectra')
    sample = np.random.default_rng(0).choice(
        len(particles), size=min(len(particles), 4096), replace=False
    )
    # Off the compute device, as in the `distance` command: this is a one-off
    # whose intermediates are larger than anything the refinement holds.
    with jax.default_device(jax.devices('cpu')[0]):
        outside_mask = mask.compute_raised_cosine_mask_2d(
            box_size=working_box,
            radius=0.5 * args.diameter / working_pixel_size,
            rolloff=max(2.0, _MASK_ROLLOFF * working_box / particles.box_size),
            inside=False
        )
        noise_spectra = noise.estimate_noise_psd_profile(
            jnp.asarray(images[np.sort(sample)]), outside_mask
        )
        projection_mask = 1.0 - outside_mask
        sigma2 = noise.estimate_projected_line_noise_variance(
            noise_spectra,
            working_box,
            key=jax.random.PRNGKey(0),
            mask=projection_mask,
        )

    # The low-pass filter attenuates both lines of a difference, so the distance
    # picks up its square as a per-frequency weight. Past the working box's own
    # Nyquist there is nothing left to filter.
    cutoff = working_pixel_size / args.resolution
    frequency_weights = None
    if cutoff < 0.5:
        frequency_weights = jnp.square(
            butterworth_1d(working_box, cutoff, order=2)
        )
    else:
        logger.info(
            'The working box band-limits below %.4g A, so no low-pass filter is '
            'applied', args.resolution,
        )

    # The CTF is a function of the sampling, so it is generated at the working
    # pixel size rather than resampled along with the images.
    working_ctf = ctf.compute_ctf_1d(
        particles.defocus,
        working_box,
        ctf.CtfContext(
            pixel_size_a=working_pixel_size,
            voltage_kv=particles.ctf_context.voltage_kv,
            spherical_aberration_mm=particles.ctf_context.spherical_aberration_mm,
            q0=particles.ctf_context.q0,
        ),
    )

    refined = refine.refine_embedding(
        embedding,
        volumes,
        images,
        rotations=particles.rotations,
        shifts=particles.shifts * (working_box / particles.box_size),
        ctf=working_ctf,
        noise_variance=sigma2,
        mask=projection_mask,
        frequency_weights=frequency_weights,
        n_epochs=args.epochs,
        anchors_per_step=args.anchors,
        partners_per_anchor=args.partners,
        ridge=args.ridge,
        huber=args.huber,
        step_size=args.step_size,
        scale=args.scale,
        noise_floor=args.noise_floor,
        floor_scale=args.floor_scale,
        partner_batch=args.partner_batch,
        anchor_batch=args.anchor_batch,
        slab=args.slab,
    )

    logger.info('Writing output')
    particles.metadata[EMBEDDING_COLUMN] = [
        '[' + ', '.join(f'{v:.4e}' for v in row) + ']'
        for row in refined
    ]
    particles.star['particles'] = particles.metadata
    starfile.write(particles.star, args.output)
