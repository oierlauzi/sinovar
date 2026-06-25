from typing import Optional

import jax
import jax.numpy as jnp

@jax.jit
def wiener_ctf_correct_1d(
    images_ft: jax.Array,
    ctfs: jax.Array,
    inv_ssnr: Optional[jax.Array] = None
) -> jax.Array:
    ctfs2 = jnp.square(ctfs)

    if inv_ssnr is None:
        inv_ssnr = 0.1 * jnp.mean(ctfs2, axis=-1, keepdims=True)
    
    return (images_ft * ctfs) / (ctfs2 + inv_ssnr)
