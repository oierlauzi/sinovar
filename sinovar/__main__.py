from typing import Sequence
import argparse
import starfile
import logging
import jax
import jax.numpy as jnp
import itertools
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from . import geometry
from . import image
from . import sinogram
from . import geometry

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
        '--prefix',
        metavar='DIR',
        help='Prefix for the MRC binary files.'
    )
    parser.add_argument(
        '--padding_factor',
        type=float,
        default=2.0,
        help='Padding factor to increase spectral resolution'
    )
    parser.add_argument(
        '--components',
        type=int,
        default=6,
        help='Number of components for dimensionality reduction'
    )
    parser.add_argument(
        '--diameter',
        type=float,
        help='Particle diameter in angstrom'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=4.0,
        help='Maximum resolution in angstrom'
    )
    parser.add_argument(
        '--angular_sampling_index',
        type=float,
        default=1.0,
        help='Angle sampling index for the sonogram generation'
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="gpu:0", 
        help="Device to use. Format: 'cpu' or 'gpu:X' (e.g., 'gpu:0', 'gpu:1')"
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
        available_backends = jax.backends()
        raise RuntimeError(
            f"Backend '{backend_name}' is not available. "
            f"Available backends: {available_backends}"
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
    """
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
    rotations = geometry.euler_zyz_to_matrix(
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
    """
    particles_md = starfile.read(args.input)
    pixel_size = 1.0
    
    image_locations = particles_md['image'].map(image.ImageLocation.parse)
    rotations = geometry.euler_zyz_to_matrix(
        jnp.deg2rad(jnp.asarray(particles_md['angleRot'])),
        jnp.deg2rad(jnp.asarray(particles_md['angleTilt'])),
        jnp.deg2rad(jnp.asarray(particles_md['anglePsi'])),
    )
    shifts = (1/pixel_size) * jnp.stack(
        (
            jnp.asarray(particles_md['shiftX']), 
            jnp.asarray(particles_md['shiftY'])
        ), 
        axis=1
    )
    
    sinogram_angles = sinogram.generate_sinogram_angles(
        diameter=args.diameter,
        resolution=args.resolution,
        sampling_ratio=args.angular_sampling_index
    )
    
    image_reader = image.BatchReader(prefix=args.prefix)
    images = jnp.asarray(image_reader.read_batch(image_locations)).astype(jnp.float32)
    
    for i, j in itertools.combinations(range(len(images)), r=2):
        angle0, angle1 = geometry.compute_intrinsic_common_line_angles(
            rotations[i],
            rotations[j]
        )
        
        common_line0 = sinogram.project_images(images[None,i], shifts[None,i], angle0[None])
        common_line1 = sinogram.project_images(images[None,j], shifts[None,j], angle1[None])
        
        plt.plot(common_line0[0])
        plt.plot(common_line1[0])
        plt.show()

def main():
    args = _parse_args()
    run(args)
    