import argparse
import logging
import random

import jax
import numpy as np

from .. import distance
from .. import image
from .. import mask
from .. import noise
from .common import add_device_argument, read_particles, select_devices

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
    add_device_argument(parser)


def run(args: argparse.Namespace) -> None:
    logger.info('Reading input')
    particles = read_particles(args.input)
    pixel_size = particles.pixel_size
    box_size = particles.box_size
    image_locations = particles.locations
    image_count = len(particles)
    devices = select_devices(args.device)

    image_reader = image.BatchReader(prefix=args.prefix)

    logger.info('Estimating noise spectra')
    noise_estimation_sample_size = min(image_count, 4096)
    rng = random.Random(0)
    image_location_selection = rng.sample(
        image_locations.tolist(),
        noise_estimation_sample_size
    )
    image_selection = image_reader.read_batch(
        image_location_selection, dtype=np.float32
    )

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
        image_reader=image_reader,
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
