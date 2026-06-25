from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_args=('diameter', 'resolution', 'sampling_ratio'))
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
