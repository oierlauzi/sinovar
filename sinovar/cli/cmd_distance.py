import argparse
import logging
import random

import jax
import numpy as np
import vitrio as vio

from .. import distance
from .. import image
from .. import mask
from .. import noise
from .common import (add_device_argument, parse_size, read_particles,
                     select_devices)

logger = logging.getLogger(__name__)

COMMAND = 'distance'
HELP = 'Compute the pairwise squared-distance matrix between particles'


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='STAR',
        help='Input STAR file with particle data'
    )
    parser.add_argument(
        '-d', '--distance',
        required=True,
        metavar='NPY',
        help='Output distance matrix.'
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
        '--block_size',
        type=int,
        default=64,
        help='Number of particles per distance-matrix tile dimension'
    )
    parser.add_argument(
        '--scratch',
        type=str,
        default='none',
        metavar='none|memory|PATH',
        help=image.SCRATCH_HELP,
    )
    parser.add_argument(
        '--scratch_budget',
        type=parse_size,
        default=None,
        metavar='BYTES',
        help='Largest the scratch may grow, as a plain number or with a K, M, '
             'G or T suffix. Required for a directory scratch, optional '
             'otherwise, where it defaults to holding every particle.'
    )
    add_device_argument(parser)


def run(args: argparse.Namespace) -> None:
    logger.info('Reading input')
    particles = read_particles(args.input, args.prefix)
    pixel_size = particles.pixel_size
    box_size = particles.box_size
    image_locations = particles.locations
    image_count = len(particles)
    devices = select_devices(args.device)

    # vitrio owns the reader pool and the optional scratch. The scratch is
    # built over every location, since the tiled sweep reads each particle
    # once per tile row, which is what makes a second copy worth keeping.
    scratch = image.open_scratch(
        args.scratch, image_locations,
        run_length=args.block_size, budget=args.scratch_budget,
    )
    image_loader = vio.loader(scratch=scratch)

    logger.info('Estimating noise spectra')
    noise_estimation_sample_size = min(image_count, 4096)
    rng = random.Random(0)
    image_location_selection = rng.sample(
        image_locations, noise_estimation_sample_size
    )
    image_selection = image.read_batch(image_loader, image_location_selection)

    with jax.default_device(jax.devices('cpu')[0]):
        outside_mask = mask.compute_raised_cosine_mask_2d(
            box_size=box_size,
            radius=0.5*args.diameter/pixel_size,
            rolloff=16,
            inside=False
        )
        noise_spectra = noise.estimate_noise_psd_profile(
            image_selection,
            outside_mask
        )
        projection_mask = 1.0 - outside_mask
        sigma2 = noise.estimate_projected_line_noise_variance(
            noise_spectra,
            int(box_size),
            key=jax.random.PRNGKey(0),
            mask=projection_mask,
        )

    mmap_distances2 = np.lib.format.open_memmap(
        args.distance,
        dtype=np.float32,
        shape=(image_count, image_count),
        mode='w+'
    )

    logger.info('Computing distance matrix')
    cutoff = pixel_size / args.resolution
    if cutoff >= 0.5:
        cutoff = None
    streaming_distance_matrix = distance.StreamingSquaredDistanceMatrix(
        loader=image_loader,
        image_locations=image_locations,
        rotations=particles.rotations,
        shifts=particles.shifts,
        defocus=particles.defocus,
        box_size=box_size,
        ctf_context=particles.ctf_context,
        sigma2=sigma2,
        mask=projection_mask,
        devices=devices,
        block_size=args.block_size,
        low_pass_cutoff=cutoff
    )
    streaming_distance_matrix.run(out=mmap_distances2)
    mmap_distances2.flush()
