from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=('order', ))
def butterworth_1d(box_size: int, cutoff: float, order: int):
    cutoff2 = cutoff*cutoff
    k = jnp.fft.rfftfreq(box_size)
    k2 = jnp.square(k2)
    k2_cutoff2 = k2 / cutoff2
    
    if order == 1:
        term = k2_cutoff2
    elif order == 2:
        term = k2_cutoff2*k2_cutoff2
    else:
        term = jnp.pow(k2_cutoff2, order)
    return 1.0 / (1.0 + term)
