import jax
import jax.numpy as jnp

def generate_sinogram_angles(
    diameter: float,
    resolution: float,
    sampling_ratio: float = 1.0
) -> jax.Array:
    shannon_angle = resolution / diameter
    sampling_step = shannon_angle * sampling_ratio
    count = round(jnp.pi / sampling_step)
    step = jnp.pi / count
    return step* jnp.arange(count)
