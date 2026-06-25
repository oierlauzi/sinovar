import jax
import jax.numpy as jnp

def _index_single_sinogram_fourier(
    sinogram: jax.Array,
    index: jax.Array
) -> jax.Array:
    L = sinogram.shape[0]
    wrapped = jnp.mod(index, 2*L)
    mirror = wrapped >= L
    row = jnp.where(mirror, wrapped - L, wrapped)
    sample = sinogram[row]
    return jnp.where(mirror, jnp.conj(sample), sample)

def _sample_single_sinogram_fourier(
    sinogram: jax.Array,
    angle: jax.Array
) -> jax.Array:
    L = sinogram.shape[0]

    pos = angle * L / jnp.pi
    lo = jnp.floor(pos).astype(jnp.int32)
    frac = pos - lo

    lower = _index_single_sinogram_fourier(sinogram, lo)
    upper = _index_single_sinogram_fourier(sinogram, lo+1)
    return (1.0 - frac)*lower + frac*upper

project_sinogram = jax.jit(
    jax.vmap(_sample_single_sinogram_fourier, in_axes=(0, 0))
)
