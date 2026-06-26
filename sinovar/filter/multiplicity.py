from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=('length', ))
def rfft_multiplicity(length: int) -> jax.Array:
    result = jnp.full(length//2 + 1, fill_value=2)
    result = result.at[0].set(1)
    
    if length % 2 == 0:
        result = result.at[-1].set(1)
    
    return result
