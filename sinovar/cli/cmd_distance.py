import argparse
import logging
import random

import jax
import jax.numpy as jnp
import numpy as np
import starfile

from .. import ctf
from .. import distance
from .. import geometry
from .. import image
from .. import mask
from .. import noise
from .common import add_device_argument, select_devices

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
    star = starfile.read(args.input)
    particles_md = star['particles']
    optics = star['optics']
    pixel_size = optics.at[0, 'rlnImagePixelSize']
    amplitude_contrast = optics.at[0, 'rlnAmplitudeContrast']
    spherical_aberration = optics.at[0, 'rlnSphericalAberration']
    voltage = optics.at[0, 'rlnVoltage']
    box_size = optics.at[0, 'rlnImageSize']
    devices = select_devices(args.device)

    image_locations = particles_md['rlnImageName'].map(image.ImageLocation.parse)
    matrices = geometry.euler_zyz_to_matrix(
        jnp.deg2rad(jnp.asarray(particles_md['rlnAngleRot'])),
        jnp.deg2rad(jnp.asarray(particles_md['rlnAngleTilt'])),
        jnp.deg2rad(jnp.asarray(particles_md['rlnAnglePsi'])),
    )
    shifts = (1/pixel_size) * jnp.stack(
        (
            jnp.asarray(particles_md['rlnOriginXAngst']),
            jnp.asarray(particles_md['rlnOriginYAngst'])
        ),
        axis=1
    )
    defocus_u = jnp.asarray(particles_md['rlnDefocusU'])
    defocus_v = jnp.asarray(particles_md['rlnDefocusV'])
    defocus = 0.5*(defocus_u + defocus_v)
    image_count = len(image_locations)

    image_reader = image.BatchReader(prefix=args.prefix)
    ctf_context = ctf.CtfContext(
        pixel_size_a=pixel_size,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        q0=amplitude_contrast
    )

    logger.info('Estimating noise spectra')
    noise_estimation_sample_size = min(image_count, 4096)
    rng = random.Random(0)
    image_location_selection = rng.sample(
        image_locations.tolist(),
        noise_estimation_sample_size
    )
    image_selection = image_reader.read_batch(image_location_selection)

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
        rotations=matrices,
        shifts=shifts,
        defocus=defocus,
        box_size=box_size,
        ctf_context=ctf_context,
        sigma2=sigma2,
        mask=projection_mask,
        devices=devices,
        block_size=args.block_size,
        low_pass_cutoff=cutoff
    )
    streaming_distance_matrix.run(out=mmap_distances2)
    mmap_distances2.flush()
