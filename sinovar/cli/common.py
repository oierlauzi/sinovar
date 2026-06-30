import argparse
import logging

import jax

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)


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
