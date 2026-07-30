from dataclasses import dataclass
from typing import Any
import argparse
import logging

import jax
import jax.numpy as jnp
import pandas as pd
import starfile

from .. import ctf
from .. import geometry
from .. import image

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)


@dataclass
class Particles:
    """A particle STAR file, with its geometry and optics in array form."""

    #: The whole file, so that a command can add a column and write it back.
    star: Any
    metadata: pd.DataFrame
    pixel_size: float
    box_size: int
    ctf_context: ctf.CtfContext
    #: Where each particle's image lives, in the file's own order.
    locations: pd.Series
    #: Orientation matrices, ``(N, 3, 3)``.
    rotations: jax.Array
    #: Origin shifts in pixels, ``(N, 2)``, ordered ``(x, y)``.
    shifts: jax.Array
    #: Mean defocus in angstrom, ``(N,)``.
    defocus: jax.Array

    def __len__(self) -> int:
        return len(self.locations)


def read_particles(path: str) -> Particles:
    """Read a particle STAR file into the arrays every command works from."""
    star = starfile.read(path)
    metadata = star['particles']
    optics = star['optics']
    pixel_size = optics.at[0, 'rlnImagePixelSize']

    rotations = geometry.euler_zyz_to_matrix(
        jnp.deg2rad(jnp.asarray(metadata['rlnAngleRot'])),
        jnp.deg2rad(jnp.asarray(metadata['rlnAngleTilt'])),
        jnp.deg2rad(jnp.asarray(metadata['rlnAnglePsi'])),
    )
    shifts = (1 / pixel_size) * jnp.stack(
        (
            jnp.asarray(metadata['rlnOriginXAngst']),
            jnp.asarray(metadata['rlnOriginYAngst'])
        ),
        axis=1
    )
    defocus = 0.5 * (
        jnp.asarray(metadata['rlnDefocusU'])
        + jnp.asarray(metadata['rlnDefocusV'])
    )

    return Particles(
        star=star,
        metadata=metadata,
        pixel_size=pixel_size,
        box_size=int(optics.at[0, 'rlnImageSize']),
        ctf_context=ctf.CtfContext(
            pixel_size_a=pixel_size,
            voltage_kv=optics.at[0, 'rlnVoltage'],
            spherical_aberration_mm=optics.at[0, 'rlnSphericalAberration'],
            q0=optics.at[0, 'rlnAmplitudeContrast'],
        ),
        locations=metadata['rlnImageName'].map(image.ImageLocation.parse),
        rotations=rotations,
        shifts=shifts,
        defocus=defocus,
    )


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--device',
        type=str,
        nargs='+',
        default=['gpu:0'],
        metavar='DEVICE',
        help="Device(s) to use. Format: 'cpu' or 'gpu:X' (e.g., 'gpu:0', "
             "'gpu:1'). Pass several to distribute the computation."
    )


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


def select_devices(indices):
    return [select_device(index) for index in indices]
