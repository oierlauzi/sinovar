from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_args=('matrices', 'norm'))
def find_common_lines(
    a_directions: jax.Array,
    b_directions: jax.Array,
    matrices: bool = True,
    norm: bool = False,
) -> jax.Array:
    if matrices:
        a_directions = a_directions[...,2,:]
        b_directions = b_directions[...,2,:]
    
    cross = jnp.linalg.cross(a_directions, b_directions)

    if norm:
        cross = cross / jnp.linalg.norm(cross, axis=-1, keepdims=True)
    
    return cross

@jax.jit
def find_common_line_angles_in_images(
    matrices: jax.Array,
    common_line_direction: jax.Array
) -> jax.Array:
    vx = matrices[...,0,:]
    vy = matrices[...,1,:]

    px = jnp.vecdot(common_line_direction, vx, axis=-1)
    py = jnp.vecdot(common_line_direction, vy, axis=-1)
    
    return jnp.atan2(py, px)
    