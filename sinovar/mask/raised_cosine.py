from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=('box_size', 'radius', 'rolloff', 'inside'))
def compute_raised_cosine_mask_2d(
    box_size: int, 
    radius: float, 
    rolloff: float, 
    inside: bool = True
) -> jax.Array:
    t = jnp.arange(box_size) - box_size // 2
    x = t[None,:]
    y = t[:,None]
    r = jnp.hypot(x, y)
    theta = jnp.pi*jnp.clip((r - radius) / rolloff, 0.0, 1.0)
    v = jnp.cos(theta)
    
    if inside:
        v = -v
    
    return 0.5*(1.0 - v)
