import argparse
import starfile
import logging
import jax
import jax.numpy as jnp
import numpy as np
import random

logger = logging.getLogger(__name__)

from . import geometry
from . import image
from . import embedding
from . import geometry
from . import distance
from . import ctf
from . import noise
from . import mask


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='sinovar',
        description='Heterogeneity analysis pipeline for CryoEM',
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='STAR',
        help='Input STAR file with particle data'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        metavar='STAR',
        help='Output STAR file with embedding data'
    )
    parser.add_argument(
        '-d', '--distance',
        required=True,
        metavar='NPY',
        help='Output distance matrix. May help RAM issues by memory-mapping.'
    )
    parser.add_argument(
        '--prefix',
        metavar='DIR',
        help='Prefix for the MRC binary files.'
    )
    parser.add_argument(
        '--components',
        type=int,
        default=6,
        help='Number of components for dimensionality reduction'
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
        "--device",
        type=str,
        default="gpu:0",
        help="Device to use. Format: 'cpu' or 'gpu:X' (e.g., 'gpu:0', 'gpu:1')"
    )
    parser.add_argument(
        '--block_size',
        type=int,
        default=64,
        help='Number of particles per distance-matrix tile dimension'
    )

    return parser.parse_args(argv)

def select_device(index: str):
    try:
        if ":" in index:
            backend_name, device_id = index.split(":")
            device_id = int(device_id)
        else:
            backend_name = index
            device_id = 0  # Default to first device if no ID provided
            
        backend_name = backend_name.lower()
    except ValueError:
        raise ValueError(f"Invalid device format: '{index}'. Use 'cpu' or 'gpu:N'.")

    try:
        available_devices = jax.devices(backend_name)
        target_device = available_devices[device_id]
        logger.info(f"Successfully selected device: {target_device}")
        
    except RuntimeError:
        raise RuntimeError(
            f"Backend '{backend_name}' is not available. "
        )
    except IndexError:
        num_devices = len(jax.devices(backend_name))
        raise IndexError(
            f"Device ID {device_id} out of bounds for backend '{backend_name}'. "
            f"Found only {num_devices} device(s)."
        )
        
    return target_device

def run(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    
    logger.info('Reading input')
    star = starfile.read(args.input)
    particles_md = star['particles']
    optics = star['optics']
    pixel_size = optics.at[0, 'rlnImagePixelSize']
    amplitude_contrast = optics.at[0, 'rlnAmplitudeContrast']
    spherical_aberration = optics.at[0, 'rlnSphericalAberration']
    voltage = optics.at[0, 'rlnVoltage']
    box_size = optics.at[0, 'rlnImageSize']
    device = select_device(args.device)
    
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
        # Project through the complement of the noise mask: it keeps the particle
        # and drops the box corners, so every projection angle shares the same
        # circular support (no angle-dependent clipping) and the signal-free
        # corners --- the very region the noise was measured on --- stop diluting
        # the common line.
        projection_mask = 1.0 - outside_mask
        # The distance compares 1D common-line projections, whose interpolation
        # reshapes and attenuates the 2D noise PSD. Calibrate the per-frequency
        # projected-line noise variance by pushing it through the same projector
        # and mask, so it matches what the distance kernel sees.
        sigma2 = noise.estimate_projected_line_noise_variance(
            noise_spectra,
            int(box_size),
            key=jax.random.PRNGKey(0),
            mask=projection_mask,
        )

    mmap_distances2 = None
    if args.distance is not None:
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
    distance_matrix = distance.StreamingSquaredDistanceMatrix(
        image_reader=image_reader,
        image_locations=image_locations,
        rotations=matrices,
        shifts=shifts,
        defocus=defocus,
        box_size=box_size,
        ctf_context=ctf_context,
        sigma2=sigma2,
        mask=projection_mask,
        devices=[device],
        block_size=args.block_size,
        low_pass_cutoff=cutoff
    )
    distances2 = distance_matrix.run(out=mmap_distances2)
    
    if mmap_distances2 is not None:
        mmap_distances2.flush()

    logger.info('Computing affinity matrix from distances')
    sigma2 = embedding.adaptive_sigma2_median(distances2, k=16)
    affinity = embedding.knn_affinity_from_squared_distance_matrix(
        distances2,
        k=4096,
        sigma2=sigma2
    )
    
    logger.info('Computing the embedding')
    y = embedding.compute_diffusion_embedding(
        affinity, 
        n_components=args.components
    )

    logger.info('Writing output')
    particles_md['sinovarEmbedding'] = [
        '[' + ', '.join(f'{v:.4e}' for v in row) + ']'
        for row in y
    ]
    star['particles'] = particles_md
    starfile.write(star, args.output)



def main():
    args = _parse_args()
    run(args)
    